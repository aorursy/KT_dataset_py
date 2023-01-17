import numpy as np

import random

import matplotlib.pyplot as plt

import time

from typing import Tuple, Optional, Dict



# DEAP

from deap import base, creator, tools, algorithms



# OR-TOOLS

from ortools.linear_solver import pywraplp



np.random.seed(seed=42)

random.seed(a=42)
def fitness_of(genotype: np.ndarray) -> float:

    """

    Evaluates the fitness of a genotype.

    

    If a genotype is not feasible (excedes the capacity of the knapsack)

    it fitness is zero.

    

    Note: maybe better penalize the violation so that a 

    "slightly" infieasible solution is not equal to a 

    higly infeasible solution.

    

    :param genotype:

        the genotype for which you want to evaluate the fitness

        

    :return: the fitness value of the genotype

    """

    is_feasible = (genotype*weights).sum() <= capacity

    if is_feasible:    

        return (genotype*values).sum()

    else:

        return 0.0
def cross_over(father: np.ndarray, mother: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """

    Implements a One Point Crossover over two individuals.

    

    Generates two sons, with inverted heredity

    

    :param father:

        the first individual to cross

    :param mother:

        the second individual to cross

        

        

    :return: first son(father|mother), second son(mother|father), when crossover happened

    """

    cross_point = np.random.randint(len(father)+1)

    son1 = np.append(father[:cross_point], mother[cross_point:])

    son2 = np.append(mother[:cross_point], father[cross_point:])



    return son1, son2, cross_point
def selection(population: np.ndarray, fitness: np.ndarray, taboo:Optional[int]=None) -> int: 

    """

    Implements the Roulette Wheel Selection over a population.

    

    :param population:

        the population from which select the individual

    :param fitness:

        the fitness of each individual in the population

    :param taboo:

        if not None, an index of an individual that can not be selected

    

    :return: the index of the indivual selected

    """

    f_copy = np.copy(fitness)

    f_copy = np.clip(f_copy, 0, None)

    if taboo is not None:

        f_copy[taboo] = 0

    r = np.random.uniform(0, f_copy.sum())    

    for x in range(len(fitness)):

        if f_copy[:x+1].sum() > r:

            return x

    

    # Pick a random individual

    return random.sample([i_ for i_ in range(len(population)) if taboo == None or not i_ == taboo], 1)[0]
def replace_individual(x:int, new_ind: np.ndarray, population: np.ndarray, age: np.ndarray, fitness: np.ndarray):

    """

    Replaces an individual of a population with a new one.

    

    Kind of simulate the death of a individual.

    

    :param x:

        the position in the population of the individual to replace

    :param new_ind:

        the new genotype to welcome in the population

    :param population:

        the population in which the replacement must be done

    :param age:

        the age of each individual in the population

    :param fitness:

        the fitness of each individual in the population

    """

    population[x] = new_ind

    age[x] = 0

    fitness[x] = fitness_of(population[x])
def mutate_drop(genotype: np.ndarray) -> np.ndarray:

    """

    Mutation that drops a item from the knapsack.

    

    Tends to fix infeasibility issues.

    

    Note: it can be implemented in a more efficient way by avoiding

    the calculation of fitness_function in the while condition.

    

    :param genotype:

        a genotype to mutate

        

    :return: 

        the mutated genotype

    """

    while fitness_of(genotype) <= 0.0001 and genotype.sum() > 0:        

        x_ = random.sample(list(np.argwhere(genotype == 1)), 1)[0]

        genotype[x_] = 0

    return genotype 



def mutate_pop(genotype: np.ndarray) -> np.ndarray:

    """

    Mutation that adds item to the genotype.

    

    If population has very few items, crossovoer will have hard

    time in increasing this number. This mutation may help newborn to have

    more items in their knapsack.

    

    This is similar to the improve function of

    [Gunther R. Raidl, An Improved Genetic Algorithm for the Multiconstrained

    0–1 Knapsack Problem]. 

    

    Note: it can be implemented in a more efficient way by avoiding

    the calculation of fitness_function in the while condition.

    

    :param genotype:

        a genotype to mutate

        

    :return: 

        the mutated genotype

    """

    def small_enough_items(genotype) -> list():

        """ Gets the items that are small enough to be added to a solution """

        residual_cap = capacity - (genotype*weights).sum()

        return np.argwhere((weights < residual_cap) & (genotype == 0))

    

    smallies = small_enough_items(genotype)

    while len(smallies) > 0:        

        x_ = random.sample(list(smallies), 1)[0]

        genotype[x_] = 1

        smallies = small_enough_items(genotype)

            

    return genotype 

    

    

    if genotype.sum() < len(genotype):

        genotype[random.sample(list(np.argwhere(genotype == 0)), 1)[0]] = 1

    return genotype 



def mutate_switch(genotype: np.ndarray, likelihood: np.ndarray= None) -> np.ndarray:

    """

    Mutation that pops an item and adds an another one. Pretty much 

    similar to Bit Flip Mutation

    

    If population has very few items, crossovoer will have hard

    time in increasing this number. This mutation may help.

    

    This is a repair function similar to the one in 

    [Gunther R. Raidl, An Improved Genetic Algorithm for the Multiconstrained

    0–1 Knapsack Problem]. 

    

    Note: it can be implemented in a more efficient way by avoiding

    the calculation of fitness_function in the while condition.

    

    :param genotype:

        a genotype to mutate

    :param likelihood:

        (optional) a numpy array stating the likelihood of each gene to be in the 

        optimal solution, if None a random flip mutation will be actuated

        

    :return: 

        the mutated genotype

    """

    if genotype.sum() > 0 and genotype.sum() < len(genotype):

        items_in = list(np.argwhere(genotype == 1))

        items_out = list(np.argwhere(genotype == 0))

                             

        if likelihood is None:

            from_x = random.sample(items_in, 1)[0]

            to_x = random.sample(items_out, 1)[0]

        else:

            # Higher probability to remove items with less likelihood

            from_x = selection(items_in, np.take(1/likelihood, items_in))

            # Higher probability to insert items with more likelihood

            to_x = selection(items_out, np.take(likelihood, items_out))

        

        to_gene = genotype[to_x]

        #print(f"{from_x}-{to_x}")

        genotype[to_x] = genotype[from_x]

        genotype[from_x] = to_gene



    return genotype
def print_generation(generation, population, fitness):

    print(f"--- Population at generation {str(generation)} ---")

    for i_ in range(len(population)):

        print(f"{i_}: {population[i_]} => {fitness[i_]}")
def GA(population_0: np.ndarray, n_iterations: int, 

       mutation_p: float, mates_p: float,       

       do_pop_mutation: bool,

       pop_max_age: int,

       do_drop_mutation: bool,

       do_switch_mutation: bool,

       do_weighted_switch_mutation: bool,

       print_evolution: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """

    Implements a naive genetic algorithm.

    

    Relies on the existance of some global variables (I know..., I know...):      

        capacity:

             Capacity of the knapsack

        values:

            Value associated to each item, to be maximized in the o.f.

        weights:

            Weight of each item, the sum must not exceed the capacity

        n_genes:

            Number of items that can be selected = number of genes in a individual

    

    :param population_0:

        the initial population

    :param n_iterations:

        the number of iterations to perform (the only implemented stopping criteria)

    :param mutation_p:

        the mutation probability, For example, 0.2 measn that a (feasible) individual

        have a probability of 20% to be mutated

    :param mates_p:

        the expected number of mates each generation as percentage of total population.

        E.g.: 0.2 => 20%, if population has 10 individual then on average will mate 2 

        pairs of them. Each mate generates two twins

    :param do_pop_mutation:

        Perform pop mutation

    :param pop_max_age:

        Max age for pop mutation

    :param do_drop_mutation:

        Perform drop mutation

    :param do_switch_mutation:

        Perform switch mutation    

    :param do_weighted_switch_mutation:

        Switch mutation is "weighted"

    :param print_evolution:

        if True, detailed evolution parameters are printed on-the-go

        

    :return: 

        - the fitting of the best individual of the population alive in each iteration,

        - the average fitting of a population alive in each iteration,

        - the fitting of the "chosen one", the most fitting individual existed up to a given

        iteration, for each iteration

        - the population existing at the last iteration    

    """

    population = np.copy(population_0)

    

    n_individuals = len(population)

    

    age = np.zeros(n_individuals) 



    best_of_generation = np.zeros(n_iterations)

    average_of_generation = np.zeros(n_iterations)

    chosen_one_at_gen = np.zeros(n_iterations)



    fitness = np.asarray([fitness_of(x) for x in population])

    chosen_one = None

    for i in range(n_iterations):

        #fitness = np.asarray([fitness_of(x) for x in population])

        

        if print_evolution:

            print_generation(i, population, fitness)

            

        if chosen_one is None or fitness.max() > fitness_of(chosen_one):

            chosen_one = np.copy(population[np.argmax(fitness)])



        # Update counters for plotting

        best_of_generation[i] = fitness.max()

        average_of_generation[i] = fitness.mean()

        chosen_one_at_gen[i] = fitness_of(chosen_one)



        age = age + 1 

        for mate in range(int(n_individuals*mates_p) + 1):

            if fitness.sum() > 0.00001:

                # Select who is going to reproduce itself

                father_x = selection(population, fitness)

                mother_x = selection(population, fitness, father_x)

                son1, son2, cross_point = cross_over(population[father_x], population[mother_x])



                # Select two elders that are going to die

                older1_x = selection(population, age)

                older2_x = selection(population, age, older1_x)

                

                # Replace the individuals

                replace_individual(older1_x, son1, population, age, fitness)

                replace_individual(older2_x, son2, population, age, fitness)

                

                if print_evolution:

                    print(f"Crossing: {father_x} + {mother_x} ({cross_point}) => {son1}, {son2} in {older1_x},{older2_x} ")

        

        # Mutations

        if do_weighted_switch_mutation:

            likelihood = values/weights

        

        for g in range(n_individuals):

            # Common mutation on "sane"  individuals

            if do_switch_mutation:

                if(np.random.uniform() < mutation_p): 

                    if print_evolution:

                        before = np.copy(population[g])

                        

                    if do_weighted_switch_mutation:

                        population[g] = mutate_switch(population[g], likelihood)

                    else:

                        population[g] = mutate_switch(population[g])

                        

                    fitness[g] = fitness_of(population[g])

                    if print_evolution:

                        print(f"Switch-mutation of {g}: {before} => {population[g]}")

                        

            # Mutation on newborn

            if do_pop_mutation:

                if age[g] <= pop_max_age:

                    if print_evolution:

                        before = np.copy(population[g])

                    population[g] = mutate_pop(population[g])

                    fitness[g] = fitness_of(population[g])

                    if print_evolution:

                        print(f"Pop-mutation of {g}: {before} => {population[g]}")

            

            # Mutation on infeasible individuals

            if do_drop_mutation:

                if fitness[g] <= 0.00001:

                    if print_evolution:

                        before = np.copy(population[g])

                    population[g] = mutate_drop(population[g])

                    fitness[g] = fitness_of(population[g])

                    if print_evolution:

                        print(f"Drop-mutation of {g}: {before} => {population[g]}")

                

    return best_of_generation, average_of_generation, chosen_one_at_gen, population
def run_and_plot(population_0: np.ndarray, n_iterations: int, 

                 mutation_p: float, mates_p: float,

                 do_pop_mutation: bool,

                 pop_max_age: int,

                 do_drop_mutation: bool,

                 do_switch_mutation: bool,

                 do_weighted_switch_mutation: bool,

                 subplot_rows= 1, subplot_cols=1, subplot_index=1):

    """

    Runs a GA algorithm starting from a population "0" and plot the behaviour of main kpis on a subplot

    

    :param population_0:

        the initial population

    :param n_iterations:

        the number of iterations to perform (the only implemented stopping criteria)

    :param mutation_p:

        the mutation probability, For example, 0.2 measn that a (feasible) individual

        have a probability of 20% to be mutated

    :param mates_p:

        the expected number of mates each generation as percentage of total population.

        E.g.: 0.2 => 20%, if population has 10 individual then on average will mate 2 

        pairs of them. Each mate generates two twins

    :param do_pop_mutation:

        Perform pop mutation

    :param pop_max_age:

        Max age for pop mutation

    :param do_drop_mutation:

        Perform drop mutation

    :param do_switch_mutation:

        Perform switch mutation    

    :param do_weighted_switch_mutation:

        Switch mutation is "weighted"

    :param subplot_rows:

        subplot rows in the plot

    :param subplot_cols:

        subplot columns in the plot

    :param subplot_index:

        index of this subplot in the plot

    """

    x_axes = tuple([i for i in range(n_iterations)])

    print_evolution = False

    best_of_generation, average_of_generation, chosen_one_at_gen, _ = GA(population_0, n_iterations,

                                                                         mutation_p, mates_p,

                                                                         do_pop_mutation,

                                                                         pop_max_age,

                                                                         do_drop_mutation,

                                                                         do_switch_mutation,

                                                                         do_weighted_switch_mutation,

                                                                         False)

    solution = chosen_one_at_gen[n_iterations-1]

    plt.subplot(subplot_rows, subplot_cols, subplot_index)

    plt.plot(x_axes, tuple(best_of_generation))

    plt.plot(x_axes, tuple(chosen_one_at_gen))

    plt.plot(x_axes, tuple(average_of_generation))

    plt.title(f"its:{n_iterations}, mut_p:{mutation_p}, inds:{len(population_0)}, mates:{mates_p} " 

              + f"=> {solution} : gap: {0 if opt is None else (opt - solution)*100/opt:.2f}%,"

              + f"Pop:{str(do_pop_mutation)}: {pop_max_age}, "

              + f"Drop:{str(do_drop_mutation)}, "

              + f"Switch:{str(do_switch_mutation)}/{str(do_weighted_switch_mutation)}")
# Location of instance files

def load_instance(instance_dir, instance) -> Tuple[float, np.ndarray, np.ndarray,

                                                   float, int, Dict[int, Tuple[float, float]]]:

    """

    Loads and parse a Pisinger knapsack instance

    

    :param instance_dir:

        the path to folder where instances are

    :param instance:

        the name of the instance to load from instance dir

    

    :return:

        A tuple with:

        - Capacity of the knapsack: int

        - Value associated to each item: np.ndarray

        - Weight of each item np.ndarray

        - Number of items that can be selected = number of genes in a individual: int

        - Optimal solution value: int

        - Items: DEAP input dict with the population: Dict

    

    """

    items = np.genfromtxt(instance_dir + instance + "_items.csv", delimiter=',',skip_header=1)

    info = np.genfromtxt(instance_dir + instance + "_info.csv", delimiter=',')



    # Capacity of the knapsack

    capacity = info[1, 1]



    # Value associated to each item

    values = items.astype(int)[:, 1]



    # Weight of each item

    weights = items.astype(int)[:, 2]



    # Number of items that can be selected = number of genes in a individual

    n_genes = len(values)



    # Optimal solution value

    opt = info[2, 1]

    

    # This dict is useful for DEAP

    # Create the item dictionary: item name is an integer, and value is 

    # a (weight, value) 2-uple.

    items = {}

    # Create random items and store them in the items' dictionary.

    for i in range(n_genes):

        items[i] = (weights[i], values[i])   



    

    return capacity, values, weights, opt, n_genes, items
# Load Instance

capacity, values, weights, opt, n_genes, items = load_instance(

    '../input/kp01pisinger/', 

    'knapPI_1_500_1000_1')
# Setup an initial population

population_0 = np.random.randint(2, size=[100, n_genes])
# Make plottings a little bit bigger

plt.rcParams['figure.figsize'] = [20,10]
# Execute and plot the behaviour

run_and_plot(population_0, n_iterations=200, mutation_p=0.3, mates_p=0.5, 

             do_pop_mutation=True, pop_max_age=100, do_drop_mutation=True, 

             do_switch_mutation=False,

             do_weighted_switch_mutation=False)

plt.show()
run_and_plot(population_0, n_iterations=200, mutation_p=0.3, mates_p=0.5, 

             do_pop_mutation=True, pop_max_age=100, do_drop_mutation=True, 

             do_switch_mutation=True,

             do_weighted_switch_mutation=True)

plt.show()
if False:

    best_of_generation, average_of_generation, chosen_one_at_gen, population = GA(

        population_0, n_iterations=10, mutation_p=0.2, mates_p=0.2, 

        do_pop_mutation=True, pop_max_age=100, do_drop_mutation=True, 

        do_switch_mutation=False,

        do_weighted_switch_mutation=False,

        print_evolution=True)
def optimize() -> Tuple[np.ndarray, float]:

    """

    Solves the loaded instance to optimality using general purpose solver:

    

    Relies on the existance of same global variables as GA:

        capacity:

             Capacity of the knapsack

        values:

            Value associated to each item, to be maximized in the o.f.

        weights:

            Weight of each item, the sum must not exceed the capacity

        n_genes:

            Number of items that can be selected = number of genes in a individual

    

    :return:

        - x_sol: the optimal solution

        - obj_val: the value of the optimal solution

    

    """

    program = pywraplp.Solver('KP01', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

    x = np.zeros(n_genes).astype(pywraplp.Variable)



    # Generate variables

    for i in range(n_genes):

        x[i] = program.BoolVar(f'x{str(i)}')



    # Add objective function

    program.Maximize((x*values).sum())



    # Add constraint

    program.Add((x * weights).sum() <= capacity, "c10")



    # Optimize

   

    status = program.Solve()

    



    # Get objective

    obj_val = program.Objective().Value()

    



    # Print solution

    x_sol = np.asarray([x[i].solution_value() for i in range(n_genes)])



    return x_sol, obj_val
tstart = time.time()

x_sol, obj_val = optimize()

solve_time = time.time() - tstart

print(f"Solve time={solve_time}")

print(str(obj_val))
# Create classes

creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))

creator.create("Individual", set, fitness=creator.Fitness)

    

# Set initial population

toolbox = base.Toolbox()

toolbox.register("attr_item", random.randrange, n_genes)

toolbox.register("individual", tools.initRepeat, creator.Individual, 

    toolbox.attr_item, 100)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)



# Define the evaluation function

def evalKnapsack(individual):

    weight = 0.0

    value = 0.0

    for item in individual:

        weight += items[item][0]

        value += items[item][1]

    if weight > capacity:

        return 10000, 0             # Ensure overweighted bags are dominated

    return weight, value



def cxSet(ind1, ind2):

    """Apply a crossover operation on input sets. The first child is the

    intersection of the two sets, the second child is the difference of the

    two sets.

    """

    temp = set(ind1)                # Used in order to keep type

    ind1 &= ind2                    # Intersection (inplace)

    ind2 ^= temp                    # Symmetric Difference (inplace)

    return ind1, ind2



def mutSet(individual):

    """Mutation that pops or add an element."""

    if random.random() < 0.5:

        if len(individual) > 0:     # We cannot pop from an empty set

            individual.remove(random.choice(sorted(tuple(individual))))

    else:

        individual.add(random.randrange(n_genes))

    return individual,





toolbox.register("evaluate", evalKnapsack)

toolbox.register("mate", cxSet)

toolbox.register("mutate", mutSet)

toolbox.register("select", tools.selNSGA2)
# Get best solution

def get_best_DEAP_sol(hof):

    best = None

    for i_ in hof.items:

        fit = i_.fitness.values[1]

        if best is None or fit > best:

            best = fit    

    return best
NGEN = 200  # Numbe rof generations

MU = 100  # Number of individual in initial population

LAMBDA = 100  # The number of children to produce at each generation.

CXPB = 0.7  # The probability that an offspring is produced by crossover.

MUTPB = 0.2  # The probability that an offspring is produced by mutation.



pop = toolbox.population(n=MU)

hof = tools.ParetoFront()

stats = tools.Statistics(lambda ind: ind.fitness.values)

stats.register("avg", np.mean, axis=0)

stats.register("std", np.std, axis=0)

stats.register("min", np.min, axis=0)

stats.register("max", np.max, axis=0)



final_pop = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,

                          halloffame=hof, verbose=False)
best = get_best_DEAP_sol(hof)

print(f"Best solution value is {best} : gap: {(opt - best)*100/opt:.2f}%," )
NGEN = 1800  # Perform more iterations starting from the previous results

final_pop = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,

                          halloffame=hof, verbose=False)
best = get_best_DEAP_sol(hof)

print(f"Best solution value is {best} : gap: {(opt - best)*100/opt:.2f}%," )
# Load Instance

for inst in ['knapPI_1_500_1000_1', 'knapPI_2_500_1000_1', 'knapPI_3_500_1000_1', 'knapPI_14_200_1000_1']:

    capacity, values, weights, opt, n_genes, items = load_instance(

        '../input/kp01pisinger/', 

        inst)

    

    # Mine

    population_0 = np.random.randint(2, size=[100, n_genes])    

    tstart = time.time()

    best_of_generation, average_of_generation, chosen_one_at_gen, population = GA(

        population_0, n_iterations=200, mutation_p=0.3, mates_p=0.5, 

        do_pop_mutation=True, pop_max_age=100, do_drop_mutation=True, 

        do_switch_mutation=True,

        do_weighted_switch_mutation=True,

        print_evolution=False)    

    solve_time = time.time() - tstart

    best = chosen_one_at_gen[199]    

    print(f"{inst}:\tMine:\tbest solution value is {best} : gap: {(opt - best)*100/opt:.2f}%, Solve time={solve_time:.2f}s" )    

    

    # DEAP

    NGEN = 2000  # Perform more iterations starting from the previous results

    

    toolbox.register("attr_item", random.randrange, n_genes)

    toolbox.register("individual", tools.initRepeat, creator.Individual, 

        toolbox.attr_item, 100)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    pop = toolbox.population(n=MU)

    hof = tools.ParetoFront()

    tstart = time.time()    

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,

                          halloffame=hof, verbose=False)

    solve_time = time.time() - tstart

    best = get_best_DEAP_sol(hof)

    print(f"{inst}:\tDEAP:\tbest solution value is {best} : gap: {(opt - best)*100/opt:.2f}%, Solve time={solve_time:.2f}s" )    

    

    # OR-Tools

    tstart = time.time()

    _, best = optimize()

    solve_time = time.time() - tstart

    print(f"{inst}:\tOR-TOOLS:\tbest solution value is {best} : gap: {(opt - best)*100/opt:.2f}%, Solve time={solve_time:.2f}s" )    

    

    print("-------")