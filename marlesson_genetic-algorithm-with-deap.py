import random

import numpy

import pandas as pd

from deap import algorithms

from deap import base

from deap import creator

from deap import tools

import multiprocessing

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import argparse

#from scoop import futures

import hashlib

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Probem COnstants

N_DAYS        = 100

MAX_OCCUPANCY = 300

MIN_OCCUPANCY = 125

FAMINY_SIZE   = 5000

DAYS          = list(range(N_DAYS,0,-1))





#load dataset

data        = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')

submission  = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')





# Load util variables

family_size_dict  = data[['n_people']].to_dict()['n_people']

cols              = [f'choice_{i}' for i in range(10)]

choice_dict       = data[cols].T.to_dict()



# from 100 to 1

family_size_ls  = list(family_size_dict.values())

choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]



# Computer penalities in a list

penalties_dict = {

  n: [

      0,

      50,

      50  + 9 * n,

      100 + 9 * n,

      200 + 9 * n,

      200 + 18 * n,

      300 + 18 * n,

      300 + 36 * n,

      400 + 36 * n,

      500 + 36 * n + 199 * n,

      500 + 36 * n + 398 * n

  ]

  for n in range(max(family_size_dict.values())+1)

} 
# Create a Tollbox Optmizer



# The creator is a class factory that can build new classes at run-time. It will be called with first the desired name of the new class, 

# second the base class it will inherit, and in addition any subsequent arguments you want to become attributes of your class. 

# This allows us to build new and complex structures of any type of container from lists to n-ary trees.

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

creator.create("Individual", list, fitness=creator.FitnessMin)



# Now we will use our custom classes to create types representing our individuals as well as our whole population.

toolbox = base.Toolbox()
# Attribute generator

toolbox.register("attr_int",   random.randint, 1, 100)



# Structure initializers

toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, FAMINY_SIZE)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)



pop   = toolbox.population(n=1000)



print(pop[0])

print("len: ", len(pop))
# The evaluation function is pretty simple in our example. We just need to count the number of ones in an individual.

def cost_function(prediction, family_size_ls, choice_dict, choice_dict_num, penalties_dict):

    penalty = 0



    # We'll use this to count the number of people scheduled each day

    daily_occupancy = {k:0 for k in DAYS}

    

    # Looping over each family; d is the day, n is size of that family, 

    # and choice is their top choices

    for n, c, c_dict, choice in zip(family_size_ls, prediction, list(choice_dict.values()), choice_dict_num):

        d = int(c)

        daily_occupancy[d] += n



        # Calculate the penalty for not getting top preference

        if d not in choice:

            penalty += penalties_dict[n][-1]

        else:

            penalty += penalties_dict[n][choice[d]]



    # for each date, check total occupancy

    #  (using soft constraints instead of hard constraints)

    k = 0

    for v in daily_occupancy.values():

        if (v > MAX_OCCUPANCY):

            k = k + (v - MAX_OCCUPANCY)

        if (v < MIN_OCCUPANCY):

            k = k + (MIN_OCCUPANCY - v)

    #    if k > 0:

    #        penalty += 100000000 

    penalty += 100000*k



    # Calculate the accounting cost

    # The first day (day 100) is treated special

    accounting_cost = (daily_occupancy[DAYS[0]]-125.0) / 400.0 * daily_occupancy[DAYS[0]]**(0.5)

    # using the max function because the soft constraints might allow occupancy to dip below 125

    accounting_cost = max(0, accounting_cost)

    

    # Loop over the rest of the days, keeping track of previous count

    yesterday_count = daily_occupancy[DAYS[0]]

    for day in DAYS[1:]:

        today_count = daily_occupancy[day]

        diff = abs(today_count - yesterday_count)

        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))

        yesterday_count = today_count



    penalty += accounting_cost



    return (penalty, )
toolbox.register("evaluate",   cost_function, family_size_ls=family_size_ls, choice_dict=choice_dict, 

                                                 choice_dict_num=choice_dict_num, penalties_dict=penalties_dict)

toolbox.register("mate",       tools.cxUniform, indpb=0.5)

toolbox.register("select",     tools.selTournament, tournsize=10) 

toolbox.register("mutate",     tools.mutShuffleIndexes, indpb=0.5)
ngen      = 100  # Gerations

npop      = 1000 # Population



hof   = tools.ParetoFront()

stats = tools.Statistics(lambda ind: ind.fitness.values)



# Statistics

stats.register("avg", numpy.mean, axis=0)

stats.register("std", numpy.std, axis=0)

stats.register("min", numpy.min, axis=0)

stats.register("max", numpy.max, axis=0)





# Evolution

pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=npop, lambda_=npop,

                                          cxpb=0.7,   mutpb=0.3, ngen=ngen, 

                                          stats=stats, halloffame=hof)
# Best Solution

best_solution = tools.selBest(pop, 1)[0]

print("")

print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))
import matplotlib.pyplot as plt

import seaborn as sns



# History AVG

plt.figure(figsize=(10,8))

front = np.array([(c['gen'], c['avg'][0]) for c in logbook])

plt.plot(front[:,0][1:-1], front[:,1][1:-1], "-bo", c="b")

plt.axis("tight")

plt.show()
# Export Result



submission['assigned_day']=best_solution

print(submission.head())

submission.to_csv('submission_{}.csv'.format(logbook[-1]['min'][0]))  