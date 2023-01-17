!pip install pyeasyga
from pyeasyga.pyeasyga import GeneticAlgorithm



def fitness(individual, data):

    x = int("".join(str(i) for i in individual), 2) / 10

    if 0 <= x <= 4:

        y = - ( x - 2 ) ** 2 + 4

        return y

    else:

        return -1



data = [1, 1, 1, 1, 1, 1]

ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())
def fitness(individual, data):

    x = int("".join(str(i) for i in individual), 2) / 10 - 100

    if -100 <= x <= 100:

        y = x

        return y

    else:

        return -10000000





data = [1 for _ in range(11)]



ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())
def fitness(individual, data):

    x = int("".join(str(i) for i in individual), 2) / 100 - 10

    if -10 <= x <= 10:

        y = -x**2

        return y

    else:

        return -100





data = [1 for _ in range(11)]



ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())
def fitness(individual, data):

    x = int("".join(str(i) for i in individual), 2) / 1000 - 3

    if -3 <= x <= 1:

        y = x**3 + (x - 1)**2

        return y

    else:

        return -10000





data = [1 for _ in range(12)]



ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())
def fitness(individual, data):

    x = int("".join(str(i) for i in individual), 2) / 100

    if 0 <= x <= 10:

        y = (x ** 2 - 1)*(x - 3)*(x - 6)*(x - 11)

        return y

    else:

        return -100





data = [1 for _ in range(10)]



ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())
import math



def fitness(individual, data):

    x_split = individual[0:7]

    y_split = individual[7:14]

    z_split = individual[14:]

    

    x = int("".join(str(i) for i in x_split), 2) / 10

    y = int("".join(str(i) for i in y_split), 2) / 10

    z = int("".join(str(i) for i in z_split), 2) / 10

    if 0 <= x <= 10 and 0 <= y <= 10 and 0 <= z <= math.pi :

        return x**2 - (y + 2)*(3 + math.sin(z))

    else:

        return -100





data = [1 for _ in range(19)]



ga = GeneticAlgorithm(data)

ga.fitness_function = fitness

ga.run()



print(ga.best_individual())