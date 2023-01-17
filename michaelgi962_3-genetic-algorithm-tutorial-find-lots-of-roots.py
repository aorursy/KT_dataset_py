import random 



class function():

    """ 

    creates a function with the form (x+a)^b

    """

    def __init__(self, a, b):

        self.a = a

        self.b = b

    

    def evaluateDerivFunc(self,x):

        """

        evaluates the derivative function b*(x+a)^(b-1) at x

        """

        return self.b*(x+self.a)**(self.b-1)
func_count = 100 # the total count of funcitons

funcs = [function(random.randint(-100,100),random.randint(2,20)) for i in range(func_count)]
import pandas as pd

a = [func.a for func in funcs]

b = [func.b for func in funcs]

coefs = pd.DataFrame()

coefs['a']  = a

coefs['b'] = b

coefs.hist(bins=400,figsize=(20,5))
# Ref 1: https://hub.packtpub.com/using-genetic-algorithms-for-optimizing-your-models-tutorial/

# Ref 2: https://deap.readthedocs.io/en/master/examples/ga_onemax.html

# Ref 3: https://www.kaggle.com/michaelgi962/genetic-algorithm-tutorial-find-a-root
from deap import base, creator, tools
creator.create("FitnessMin",base.Fitness, weights = (-1.0,)) # weights = (-1.0,) is used when minimizing the fitness function
creator.create("Chromosome",list,fitness = creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool",random.randint,0,1)
# We will assign 8-bit precision for each root guess

n_bit_precision = 8



# We make the length of the chromosome the product of functions and the number of bits for each function's root guess

chromo_length = len(funcs)*n_bit_precision 



# Now we create the method

toolbox.register("chromosome",tools.initRepeat,creator.Chromosome,toolbox.attr_bool,chromo_length) 
toolbox.register("population",tools.initRepeat,list,toolbox.chromosome)
# Make a chromosome parsing function

def parse_chromosome(chromosome,gene_lens):

    # chromosome: a list of 1's and 0's

    # gene_lens: a list of the number of bits in each gene

    # binary_strings: a list of the genes in binary string notation

    

    # convert chromosome from list to string

    binary_string = ''.join([str(num) for num in chromosome])    



    # get the separate genes from the chromosome

    binary_strings = [binary_string[sum(gene_lens[:i]):sum(gene_lens[:i])+gene_lens[i]] 

                            for i in range(len(gene_lens))]

    return binary_strings # return the list of binary strings for each gene
# Make the binary string to root guess value conversion function

def map_binary_to_root(binary_string):

    """

    This function converts the binary string to its equivalent integer valued root guess

    """

    integer = int(binary_string,2)

    

    # convert the binary value to its mapped value on the root guess interval: [0,...,255] -> [-100,...,100] 

    root_guess = integer*(100-(-100))/(2**8 - 1) - 100

    

    return root_guess
def evaluate_chromosome_fitness(chromosome): 

    # 5a. parse the chromosome for its genes' binary strings 

    gene_lens = [8 for i in range(len(funcs))] # the precicion for each functions root guess is 8 bits

    binary_strings = parse_chromosome(chromosome,gene_lens)

    

    # 5b. convert each binary string into the equivalent integer valued root guess

    root_guesses = [root_guess for root_guess in map(map_binary_to_root,binary_strings)]



    # 5c. evaluate the fitness each binary string (i.e. each gene)

    fitnesses = []

    for func,root_guess in zip(funcs,root_guesses):

        fitnesses.append(func.evaluateDerivFunc(root_guess))

        

    # 5d. Calculate the fitness value for the chromosome (i.e. for all of the genes)

    chromosome_fitness = sum([abs(fitness) for fitness in fitnesses]) # We just add up all abs values of each genes fitness to get the chromo fitness

            

    return chromosome_fitness,

    
toolbox.register('evaluate',evaluate_chromosome_fitness)
toolbox.register('mate',tools.cxTwoPoint)

toolbox.register('mate',tools.cxUniform,indpb = 0.5)
toolbox.register('mutate',tools.mutFlipBit,indpb = 0.5)
toolbox.register('select',tools.selTournament,tournsize = 10)
pop = toolbox.population(n = 100)
fitnesses = list(map(toolbox.evaluate,pop))

for indiv,fit in zip(pop,fitnesses):

    indiv.fitness.values = fit
CXPB = 0.5
MUTPB = 0.5
# Initialize empty lists to gather the fitness stats from each generation

gen_min = []

gen_max = []

gen_avg = []

gen_std = []



# Set number of generations

NGEN = 25

gen = 0



# Initiate the evolution

while gen < NGEN:

    gen += 1

    

    # Perform tournament selection to find the next generation of superior individuals

    offspring = toolbox.select(pop,len(pop))

    

    # Clone the Offspring

    offspring = list(map(toolbox.clone,offspring))

    

    # Perform Crossover on the offspring

    for child1, child2 in zip(offspring[0::2],offspring[1::2]):

        if random.random() < CXPB: 

            toolbox.mate(child1,child2)

            del child1.fitness.values

            del child2.fitness.values

    

    # Perform Mutation on the offspring

    for mutant in offspring:

        if random.random() < MUTPB:

            toolbox.mutate(mutant)

            del mutant.fitness.values 

            

    # Find the individuals without fitness values in the offspring 

    indiv_without_fitness = [indiv for indiv in offspring if not indiv.fitness.valid]

    

    # Evaluate new fitness values for offspring without fitness values

    new_fitnesses = list(map(toolbox.evaluate,indiv_without_fitness))

    

    # Store new fitness values respective to their individuals

    for indiv,fit in zip(indiv_without_fitness,fitnesses):

        indiv.fitness.values = fit

        

    # Replace the population with the offspring

    pop[:] = offspring

    

    # Gather all of the fitnesses into a list and print the generation's stats

    fits = [ind.fitness.values[0] for ind in pop]

    

    length = len(pop)

    mean = sum(fits)/length

    sum2 = sum(x*x for x in fits)

    std = abs(sum2/length - mean**2)*0.5

    

    print("-- Generation %i --" % gen)

    '''

    print("  Min %s" % min(fits))

    print("  Max %s" % max(fits))

    print("  avg %s" % mean)

    print("  std %s" % std)

    '''

    # Accumulate fitness statistics for each generation

    gen_min.append(min(fits))

    gen_max.append(max(fits))

    gen_avg.append(mean)

    gen_std.append(std)
import pandas as pd

import matplotlib.pyplot as plt



# Store the stats for each generation in a dataframe

## columns to store are: gen_min, gen_max, gen_avg, gen_std

gen_stats = pd.DataFrame()

gen_stats['gen_min'] = gen_min

gen_stats['gen_max'] = gen_max

gen_stats['gen_avg'] = gen_avg

gen_stats['gen_std'] = gen_std



# Look at the generation statistics dataframe

gen_stats
gen_stats.columns[[0,3]]
# Plot the generation statistics evolution

gen_stats[['gen_min']].plot()
stats = tools.Statistics(key=lambda ind: ind.fitness.values)
import numpy

stats.register("avg", numpy.mean)

stats.register("std", numpy.std)

stats.register("min", numpy.min)

stats.register("max", numpy.max)
# reinitialize the population

pop = toolbox.population(n = 100)



from deap import algorithms

# Run the premade algorithm

pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.7, ngen=75, stats=stats, verbose=False)

# Plot the minimization

pd.DataFrame(logbook)['min'].plot()