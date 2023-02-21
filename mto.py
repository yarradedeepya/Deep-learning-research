import numpy as np

class MTO:
    def __init__(self, objective_function, bounds, population_size=1, max_iter=2, alpha=0.2, beta=2.0):
        self.objective_function = objective_function
        self.bounds = bounds
        self.population_size = population_size
        self.max_iter = max_iter
        self.alpha = alpha
        self.beta = beta

    def run(self):
        dim = len(self.bounds)
        lb = [self.bounds[i][0] for i in range(dim)]
        ub = [self.bounds[i][1] for i in range(dim)]
        pop = self.initialize_population(dim)
        fit = [self.objective_function(pop[i]) for i in range(self.population_size)]
        best_index = np.argmin(fit)
        best_solution = pop[best_index]
        for i in range(self.max_iter):
            for j in range(self.population_size):
                if j == best_index:
                    continue
                x = pop[j]
                v = self.update_mother_tree(pop, j, best_index)
                u = np.random.uniform(size=dim)
                a = 2.0 * self.alpha * u - self.alpha
                r1 = np.random.uniform(size=dim)
                r2 = np.random.uniform(size=dim)
                y = v + a * np.abs(r1 * best_solution - x) - a * np.abs(r2 * x - v)
                y = np.maximum(np.minimum(y, ub), lb)
                f_y = self.objective_function(y)
                if f_y < fit[j]:
                    pop[j] = y
                    fit[j] = f_y
                    if f_y < fit[best_index]:
                        best_index = j
                        best_solution = y
        return best_solution

    def initialize_population(self, dim):
        return np.random.uniform(size=(self.population_size, dim))

    def update_mother_tree(self, pop, j, best_index):
        dim = len(self.bounds)
        x = pop[j]
        best_solution = pop[best_index]
        r = np.random.uniform(size=dim)
        v = (1.0 - r) * best_solution + r * np.mean(pop, axis=0)
        beta = self.beta * (1.0 - float(j+1) / self.population_size)
        u = np.random.uniform(size=dim)
        a = 2.0 * beta * u - beta
        y = v + a * np.abs(best_solution - x)
        return np.maximum(np.minimum(y, 1.0), 0.0)
