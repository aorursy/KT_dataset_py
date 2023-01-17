# Ref 1: https://hub.packtpub.com/using-genetic-algorithms-for-optimizing-your-models-tutorial/

# Ref 2: https://deap.readthedocs.io/en/master/examples/ga_onemax.html
# Method 1 - Generate Random n-bit Binary String

import random

from random import getrandbits # Randomly generate a n-bit integer

random.seed(a=0)
n = 8 # Number of bits in chromosome

rand_bit_string_1 = getrandbits(n) # Get random 8-bit integer

rand_bit_string_1
# Get binary string representation of 8-bit bit-string

rand_bit_string_1 = '0'*(n-len(bin(rand_bit_string_1)[2:])) + bin(rand_bit_string_1)[2:]

rand_bit_string_1
# Method 2 - Generate Random n-bit Binary String

chromosome = ''

rand_bit_string_2 = ''.join([chromosome + str(round(random.random())) for i in range(n)]) 

rand_bit_string_2
from deap import base, creator, tools

import random
creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
creator.create("Individual",list,fitness=creator.FitnessMin)
# Create the toolbox

toolbox = base.Toolbox()
# Create the boolean value generator method "attr_bool"(That generates a value of 0 or 1 to be returned upon instantiation)

toolbox.register("attr_bool",random.randint,0,1)
# Create the method to create individuals called "Individual" where each individual is a list of N generated 

# boolean values using "attr_bool"

n=8 # the number of bits in the chromosome

toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,n)
# Create the generate a method to initialize a population whos size of total individuals is declared when the

# method is instantiated

toolbox.register("population",tools.initRepeat,list,toolbox.individual)
def evalOneMin(individual):

    # Convert the individual that is a 8 bit list of 0's or 1's, into a binary string of 0's and 1's

    binary_string = ''.join(str(num) for num in individual)

    

    # Decode the value of the binary string into its equivalent value on the interval between -10 and 10 

    decoded_value = 20*(int(binary_string,2)/255) - 10



    # Return the positive size of the guessed root for f(x) = 4x^3

    return abs(4*decoded_value**3), 



# Register the evaluation function with the toolbox

toolbox.register("evaluate",evalOneMin)
# register a Two Point Crossover method called 'mate'

toolbox.register("mate",tools.cxTwoPoint)
# Create the toolbox method for a FlipBit mutation method called 'mutate' with a bitwise-independent bitflip 

# probability of 0.05

toolbox.register("mutate",tools.mutFlipBit,indpb = 0.05)
# Create the toolbox method for a tournument selection method called 'select'

toolbox.register("select",tools.selTournament,tournsize = 3)
pop = toolbox.population(n=500) # Create an initial population of size 102
# Set the probabilities so that 50% of the time crossover occurs between pairs of 

# individuals 

CXPB = 0.2
# Set the probabilities so that 20% of the time, mutation occurs to an individual 

MUTPB = 0.5
# Calculate the value of the fitness function for each individual 

fitnesses = list(map(toolbox.evaluate,pop))

for ind,fit in zip(pop,fitnesses):

    ind.fitness.values = fit # Store the fitness value for each individual 
NGEN = 25
# Initialize empty lists to store the fitness stats from each generation

gen_min = []

gen_max = []

gen_avg = []

gen_std = []



for g in range(NGEN):

    print("-- Generation %i --" % g)

    # Select the next generation of individuals

    offspring = toolbox.select(pop,len(pop))

    # Clone the selected individuals

    offspring = list(map(toolbox.clone,offspring))

    # Apply crossover on the offspring

    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        if random.random() < CXPB:

            toolbox.mate(child1, child2)

            del child1.fitness.values

            del child2.fitness.values

    # Apply mutation on the offspring

    for mutant in offspring:

        if random.random() < MUTPB:

            toolbox.mutate(mutant)

            del mutant.fitness.values

            

    # Evaluate the individuals with an invalid fitness (i.e. the individuals that underwent crossover or mutation)

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

    fitnesses = map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):

        ind.fitness.values = fit

    

    # Replace the population with the offspring

    pop[:] = offspring

    

    # Gather all the fitnesses in one list and print the stats

    fits = [ind.fitness.values[0] for ind in pop]

        

    length = len(pop)

    mean = sum(fits) / length

    sum2 = sum(x*x for x in fits)

    std = abs(sum2 / length - mean**2)**0.5

        

    print("  Min %s" % min(fits))

    print("  Max %s" % max(fits))

    print("  Avg %s" % mean)

    print("  Std %s" % std)

    

    gen_min.append(min(fits))

    gen_max.append(max(fits))

    gen_avg.append(mean)

    gen_std.append(std)
best_ind = tools.selBest(pop,1)[0]

best_ind.fitness.values
import pandas as pd

import matplotlib.pyplot as plt
# Store the stats for each generation in a dataframe

## columns to store are: gen_min, gen_max, gen_avg, gen_std

gen_stats = pd.DataFrame()

gen_stats['gen_min'] = gen_min

gen_stats['gen_max'] = gen_max

gen_stats['gen_avg'] = gen_avg

gen_stats['gen_std'] = gen_std
gen_stats
gen_stats.plot(figsize=(10,10),fontsize=20)

plt.rcParams['legend.title_fontsize'] = 4