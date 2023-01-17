# Ref 1: https://hub.packtpub.com/using-genetic-algorithms-for-optimizing-your-models-tutorial/

# Ref 2: https://deap.readthedocs.io/en/master/examples/ga_onemax.html

# Ref 3: https://www.kaggle.com/michaelgi962/genetic-algorithm-tutorial-find-a-root
from deap import base, creator, tools

import random 
creator.create("FitnessMin",base.Fitness, weights = (-1.0,))
creator.create("Individual",list,fitness = creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_bool",random.randint,0,1)
x_guess_bit_len = 8

y_guess_bit_len = 10

gene_lens = [x_guess_bit_len, y_guess_bit_len]

n = x_guess_bit_len + y_guess_bit_len # the length of the chromosome is the sum of the bits from each of the two root guesses

toolbox.register("individual",tools.initRepeat,creator.Individual,toolbox.attr_bool,n)
toolbox.register("population",tools.initRepeat,list,toolbox.individual)
# parse a sample chromosome

chromo = '0110001111' # An example chromosome with extra integers 0 to 9 for visualization purposes

param_lens = [1,2,3,4] # The number of bits in each param

param_parsed = ['0','11','000','1111'] # The correctly parsed chromosome

parsed_chromo = [chromo[sum(param_lens[:i]):sum(param_lens[:i])+param_lens[i]] for i in range(len(param_lens))]
# Make a chromosome parsing function

def parse_chromosome(chromosome,gene_lens):

    # chromosome: a list of 1's and 0's

    # gene_lens: a list of the number of bits in each gene

    # binary_strings: a list of the genes in binary string notation

    

    # 5a. convert to string

    binary_string = ''.join([str(num) for num in chromosome])    



    # 5b. get the separate genes from the chromosome

    binary_strings = [binary_string[sum(gene_lens[:i]):sum(gene_lens[:i])+gene_lens[i]] 

                            for i in range(len(gene_lens))]

    return binary_strings
# Make the gene evaluation function for gene_1

def evaluate_gene_1(binary_string):

    # convert the binary string to its equivalent integer

    gene = int(binary_string,2)

    # convert the binary value to its mapped value on the x interval: [0,...,255] -> [-10,...,10] 

    gene_fitness = gene*(10-(-10))/(2**(len(binary_string))-1) - 10

    return gene_fitness
# Make the gene evaluation function for gene_2

def evaluate_gene_2(binary_string):

    # convert the binary string to its equivalent integer

    gene = int(binary_string,2)

    # convert the binary value to its mapped value on the y interval: [0,...,1023] -> [-30,...,30]

    gene_fitness = gene*(30-(-30))/(2**(len(binary_string))-1) - 30

    return gene_fitness
def evalTwoMins(individual): 

    # 5a. convert to string

    binary_string = ''.join([str(num) for num in individual])

    

    # 5b. parse chromosome into parts

    binary_strings = parse_chromosome(individual,gene_lens)



    # 5c. evaluate the fitness each binary string (i.e. each gene)

    gene_1_fitness = evaluate_gene_1(binary_strings[0])

    gene_2_fitness = evaluate_gene_2(binary_strings[1])

    

    # 5d. Calculate the fitness value for the chromosome (i.e. for all of the genes)

    chromosome_fitness = (gene_1_fitness**2 + gene_2_fitness**2)**.5

            

    return chromosome_fitness,

    
toolbox.register('evaluate',evalTwoMins)
toolbox.register('mate',tools.cxTwoPoint)
toolbox.register('mutate',tools.mutFlipBit,indpb = 0.05)
toolbox.register('select',tools.selTournament,tournsize = 10)
pop = toolbox.population(n = 100)
fitnesses = list(map(toolbox.evaluate,pop))

for indiv,fit in zip(pop,fitnesses):

    indiv.fitness.values = fit
CXPB = 0.5
MUTPB = 0.2
fits = [indiv.fitness.values[0] for indiv in pop]
NGEN = 100

gen = 0

while gen < NGEN:

    gen += 1

    print("-- Generation %i --" % gen)

    

    # Perform tournament selection to find the next generation of individuals

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

    

    print("  Min %s" % min(fits))

    print("  Max %s" % max(fits))

    print("  avg %s" % mean)

    print("  std %s" % std)
best_indiv = tools.selBest(pop,1)[0]

best_indiv 
best_indiv_binary_strings = parse_chromosome(best_indiv,gene_lens)

print("-- Note: The closer the gene's fitness value is to 0, the closer the gene is to its root --")

print("The fitness value of gene 1 of the best chromosome is: %s" % evaluate_gene_1(best_indiv_binary_strings[0]))

print("The fitness value of gene 2 of the best chromosome is: %s" % evaluate_gene_2(best_indiv_binary_strings[1]))

print("")

print("-- Note: The closer the chromosomes's fitness value is to 0, the closer the gene's combined fitness values are to their roots --")

print("The fitness value of the best chromosome is: %s" % evalTwoMins(best_indiv))