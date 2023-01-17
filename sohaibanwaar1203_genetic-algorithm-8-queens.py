import random

import string



def random_individual(size):

    return [ random.randint(1, 8) for _ in range(8) ]



maxFitness = 28

def fitness(individual):

    horizontal_collisions = sum([individual.count(queen)-1 for queen in individual])/2

    diagonal_collisions = 0



    n = len(individual)

    left_diagonal = [0] * 2*n

    right_diagonal = [0] * 2*n

    for i in range(n):

        left_diagonal[i + individual[i] - 1] += 1

        right_diagonal[len(individual) - i + individual[i] - 2] += 1



    diagonal_collisions = 0

    for i in range(2*n-1):

        counter = 0

        if left_diagonal[i] > 1:

            counter += left_diagonal[i]-1

        if right_diagonal[i] > 1:

            counter += right_diagonal[i]-1

        diagonal_collisions += counter / (n-abs(i-n+1))

    

    return int(maxFitness - (horizontal_collisions + diagonal_collisions))



def probability(individual, fitness):

    return fitness(individual) / maxFitness



def random_pick(population, probabilities):

    populationWithProbabilty = zip(population, probabilities)

    total = sum(w for c, w in populationWithProbabilty)

    r = random.uniform(0, total)

    upto = 0

    for c, w in zip(population, probabilities):

        if upto + w >= r:

            return c

        upto += w

    assert False, "Shouldn't get here"

        

def reproduce(x, y):

    n = len(x)

    c = random.randint(0, n - 1)

    return x[0:c] + y[c:n]



def mutate(x):

    n = len(x)

    c = random.randint(0, n - 1)

    m = random.randint(1, n)

    x[c] = m

    return x



def genetic_queen(population, fitness):

    mutation_probability = 0.03

    new_population = []

    probabilities = [probability(n, fitness) for n in population]

    for i in range(len(population)):

        x = random_pick(population, probabilities)

        y = random_pick(population, probabilities)

        child = reproduce(x, y)

        if random.random() < mutation_probability:

            child = mutate(child)

        print_individual(child)

        new_population.append(child)

        if fitness(child) == 28: break

    return new_population



def print_individual(x):

    print("{},  fitness = {}, probability = {:.6f}"

        .format(str(x), fitness(x), probability(x, fitness)))



if __name__ == "__main__":

    population = [random_individual(8) for _ in range(100)]

    generation = 1



    while not 28 in [fitness(x) for x in population]:

        print("=== Generation {} ===".format(generation))

        population = genetic_queen(population, fitness)

        print("Maximum fitness = {}".format(max([fitness(n) for n in population])))

        generation += 1



    print("Solved in Generation {}!".format(generation-1))

    for x in population:

        if fitness(x) == 28:

            print_individual(x)