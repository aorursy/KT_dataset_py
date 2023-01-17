#! pip install deap
from deap import base, creator, tools
import numpy as np

import random

from math import sin, cos, tanh, pow, pi, exp, sqrt



numVariables = 5 



creator.create( "FitnessMin", base.Fitness , weights=(-1.0,))

creator.create( "IndividualContainer", list , fitness= creator.FitnessMin)

toolbox = base.Toolbox()

# toolbox.register( "InitialValue", np.random.uniform, -500, 500)

toolbox.register( "InitialValue", np.random.uniform, 1, 60)

toolbox.register( "indiv", tools.initRepeat, creator.IndividualContainer, toolbox.InitialValue, numVariables)

toolbox.register( "population", tools.initRepeat, list , toolbox.indiv)



def eval_deVilliers_Glasser_2_Func( indiv ):   #47  Global Min = 0

    sum = 0

    for i in range(1, 25):

        ti = 0.1 * (i - 1)

        yi = 53.81 * 1.27**ti*tanh(3.012*ti + sin(2.13*ti)) * cos(exp(0.507)*ti)

        sum += (indiv[0]*(indiv[1]**ti)*tanh(indiv[2]*ti + sin(indiv[3]*ti))*cos(ti*exp(indiv[4])) - yi)**2.0

    #return (sqrt(sum.real**2 + sum.imag**2),) 

    return (sum,)
ind = toolbox.indiv()

print(ind)
#penalty



# MIN_BOUND = np.array([-500]*numVariables)

# MAX_BOUND = np.array([500]*numVariables)



MIN_BOUND = np.array([1]*numVariables)

MAX_BOUND = np.array([60]*numVariables)



def feasible( indiv ):

    if any( indiv < MIN_BOUND) or any( indiv > MAX_BOUND):

        return False

    return True



def distance( indiv ) :

    dist = 0.0

    for i in range (len( indiv )) :

        penalty = 0

        if ( indiv [i] < MIN_BOUND[i]) : penalty = 1 - indiv [i]

        if ( indiv [i] > MAX_BOUND[i]) : penalty = indiv [i] - 60

        dist = dist + penalty

    return dist



# def distance( indiv ) :

#     dist = 0.0

#     for i in range (len( indiv )) :

#         penalty = eval_deVilliers_Glasser_2_Func(indiv)

#         dist = dist + penalty[0]

#     return dist
toolbox.register( "evaluate", eval_deVilliers_Glasser_2_Func)

toolbox.decorate( "evaluate", tools.DeltaPenalty (feasible, 10.0, distance))

#toolbox.decorate( "evaluate", tools.DeltaPenalty (feasible, 7.0))
#crossover functions



#what effect does alpha have on the crossover?

def my_cxblend_low(ind1 , ind2 ):

    alpha = 0.05

    (ind1, ind2) = tools.cxBlend(ind1, ind2, alpha)

    return ind1 , ind2



def my_cxblend_high(ind1 , ind2 ):

    alpha = 0.5

    (ind1, ind2) = tools.cxBlend(ind1, ind2, alpha)

    return ind1 , ind2



#what is considered high eta?

def my_cxsimulated_high(ind1, ind2 ):

    eta = 1

    (ind1, ind2) = tools.cxSimulatedBinary(ind1, ind2, eta)

    return ind1 , ind2



def my_cxsimulated_low(ind1, ind2 ):

    eta = 10

    (ind1, ind2) = tools.cxSimulatedBinary(ind1, ind2, eta)

    return ind1 , ind2
#just like mutUniformInt but for floats



from __future__ import division

import math

import random



from itertools import repeat

from collections import Sequence

def mutUniformFloat(individual, low, up, indpb):

    size = len(individual)

    if not isinstance(low, Sequence):

        low = repeat(low, size)

    elif len(low) < size:

        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))

    if not isinstance(up, Sequence):

        up = repeat(up, size)

    elif len(up) < size:

        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))



    for i, xl, xu in zip(range(size), low, up):

        if random.random() < indpb:

            individual[i] = random.uniform(xl, xu)

    return (individual,)
#mutation algorithms

def mut_uniform_high(individual):

    individual = mutUniformFloat(individual, 1, 60, 0.5)

    return individual



def mut_uniform_low(individual):

    individual = mutUniformFloat(individual, 1, 60, 0.05)

    return individual
#algorithm functions

from deap import algorithms



def ea2_with_stats(cxpb, mutpb,ngen):

    import numpy

      

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)

    stats.register("min", numpy.min)

    stats.register("max", numpy.max)

    

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    

    return pop, logbook, hof



def m_plus_l_with_stats(cxpb, mutpb,ngen):

    import numpy

      

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)

    stats.register("min", numpy.min)

    stats.register("max", numpy.max)

    

    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=50 ,lambda_=150 , cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    

    return pop, logbook, hof    



def m_comma_l_with_stats(cxpb, mutpb,ngen):

    import numpy

      

    pop = toolbox.population(n=200)

    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)

    stats.register("avg", numpy.mean)

    stats.register("min", numpy.min)

    stats.register("max", numpy.max)

    

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=50 ,lambda_=150 , cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=True)

    

    return pop, logbook, hof
def run_main (crossover, crosparams, mutation, mutparams, selectsize, genalg, cxpb, mutpb,ngen):

    #crossover

    if (crossover == 'blend'):

        if (crosparams == 'high'):

            toolbox.register( "mate", my_cxblend_high)

        elif (crosparams == 'low'):

            toolbox.register( "mate", my_cxblend_low)

    elif (crossover == 'simulated'):

        if (crosparams == 'high'):

            toolbox.register( "mate", my_cxsimulated_high)

        elif (crosparams == 'low'):

            toolbox.register( "mate", my_cxsimulated_low)

    #mutation

    if (mutation == 'gaussian'):

        if (mutparams == 'high'):

            toolbox.register( "mutate", tools.mutGaussian, mu = 0.5*30, sigma=1.0, indpb=0.5)

        elif (mutparams == 'low'):

            toolbox.register( "mutate", tools.mutGaussian, mu = 0.5*30, sigma=1.0, indpb=0.05)

    elif (mutation == 'uniform'):

        if (mutparams == 'low'):

            toolbox.register( "mutate", mut_uniform_low)

        elif (mutparams == 'high'):

            toolbox.register( "mutate", mut_uniform_high)

    #selection

    if (selectsize == '3'):

        toolbox.register( "select", tools.selTournament, tournsize=3)

    else:

        toolbox.register( "select", tools.selTournament, tournsize=7)

    if (genalg == 'simple'):

        pop, log, hof = ea2_with_stats(cxpb, mutpb,ngen)

        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

        return pop, log, hof

    elif (genalg == 'mplusl'):

        pop, log, hof = m_plus_l_with_stats(cxpb, mutpb,ngen)

        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

        return pop, log, hof

    elif (genalg == 'mcommal'):

        pop, log, hof = m_comma_l_with_stats(cxpb, mutpb,ngen)

        print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

        return pop, log, hof



    
eval_deVilliers_Glasser_2_Func( [53.81, 1.27, 3.012, 2.13, 0.507] )
import pandas as pd

import numpy as np
df = pd.DataFrame(columns=['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time'])
# #FOR ONE TIME RUNS ONLY



# import time

# ngens = 100

# cxpb = 0.8

# mutpb = 0.2

# goal = 0

# delta = 3.5

# crossover = 'blend'

# mutation = 'uniform'

# crosparams = 'high'

# mutparams = 'high'

# selparam = '3'

# ga = 'simple'

# operator = crossover + crosparams + mutation + mutparams + "Sel" + selparam

# summins = 0

# sumevals = 0

# timeofrnds = 0

# successes = 0

# sumgens = 0

# sumsucgenmin = 0

# sumsucgenevals = 0

# for rnd in range (0,5):

#     start = time.time()

#     pop, log, hof = run_main (crossover, crosparams, mutation, mutparams, selparam, ga, cxpb, mutpb, ngens)

#     endtime = time.time()

#     flag = False

#     for i in range (0, ngens + 1):  #till number of gens + 1

#         sumevals += (log[i])['nevals']

#         if ((log[i]['min'] < goal + delta) & (flag == False)):

#             sumgens += log[i]['gen']

#             flag = True

#             sumsucgenmin += log[i]['min']

#             sumsucgenevals += sumevals

#     summins += hof[0].fitness.values[0]

#     if  hof[0].fitness.values[0] < goal + delta:

#         successes += 1

#     timeofrnds += endtime - start

# avgevals = sumevals/5

# avgtime = timeofrnds/5

# avgmins = summins/5

# avgsucgens = sumgens/5

# avgsumsucgenmin = sumsucgenmin/5

# avgsumsucgenevals = sumsucgenevals/5

# #df = df.append({'operators':operator, 'strategy':ga , 'successes':successes,'s.avg.min':avgsumsucgenmin, 's.avg.evals':avgsumsucgenevals, 's.avg.gens':avgsucgens, 'avg.evals':avgevals, 'avg.min':avgmins, 'avg.time':avgtime},ignore_index=True)

# print("Average number of evaluations:", avgevals)

# print("Average time of each round:" ,avgtime)

# print("Average value of fitness:", avgmins)

# print("Number of successes:", successes)

# if (avgsucgens == 0):

#     print("Average successful generation:", 'none')

# else:

#     print("Average successful generation:", avgsucgens)

# if (sumsucgenmin == 0):

#     print("Average minimum of successful generations:", 'none')

# else:

#     print("Average minimum of successful generations:", avgsumsucgenmin)

# if (sumsucgenevals == 0):

#     print("Average number of evaluations until successful generation:", 'none')

# else:

#     print("Average number of evaluations until successful generation:", avgsumsucgenevals)
#function run_main (crossover, crosparams, mutation, mutparams, selectsize, genalg)

#crossover possible values: 'blend', 'simulated'

#mutation possible values: 'gaussian', 'uniform'

#crosparams-mutparams possible values: 'low', 'high'

#genalg possible values: 'simple', 'mplusl', 'mcommal'

import time

ngens = 100

cxpb = 0.8

mutpb = 0.2

goal = 0

delta = 3.5

crossover = ['blend', 'simulated']

mutation = ['gaussian', 'uniform']

crosparams = ['low', 'high']

mutparams = ['low','high']

selparam = ['3', '7']

ga = ['simple', 'mplusl', 'mcommal']

for c in range (0,len(crossover)):

    for cp in range (0,len(crosparams)):

        for m in range (0,len(mutation)):

            for mp in range (0, len(mutparams)):

                for s in range (0, len(selparam)):

                    for al in range (0, len(ga)):

                        operator = crossover[c] + crosparams[cp] + mutation[m] + mutparams[mp] + "Sel" + selparam[s]

                        summins = 0

                        sumevals = 0

                        timeofrnds = 0

                        successes = 0

                        sumgens = 0

                        sumsucgenmin = 0

                        sumsucgenevals = 0

                        for rnd in range (0,5):

                            start = time.time()

                            pop, log, hof = run_main (crossover[c], crosparams[cp], mutation[m], mutparams[mp], selparam[s], ga[al], cxpb, mutpb, ngens)

                            endtime = time.time()

                            flag = False

                            for i in range (0, ngens + 1):  #till number of gens + 1

                                sumevals += (log[i])['nevals']

                                if ((log[i]['min'] < goal + delta) & (flag == False)):

                                    sumgens += log[i]['gen']

                                    flag = True

                                    sumsucgenmin += log[i]['min']

                                    sumsucgenevals += sumevals

                            summins += hof[0].fitness.values[0]

                            if  hof[0].fitness.values[0] < goal + delta:

                                successes += 1

                            timeofrnds += endtime - start

                        avgevals = sumevals/5

                        avgtime = timeofrnds/5

                        avgmins = summins/5

                        avgsucgens = sumgens/5

                        avgsumsucgenmin = sumsucgenmin/5

                        avgsumsucgenevals = sumsucgenevals/5

                        df = df.append({'operators':operator, 'strategy':ga[al] , 'successes':successes,'s.avg.min':avgsumsucgenmin, 's.avg.evals':avgsumsucgenevals, 's.avg.gens':avgsucgens, 'avg.evals':avgevals, 'avg.min':avgmins, 'avg.time':avgtime},ignore_index=True)
display(df)
# #GRIDSEARCH

# gsdf = pd.DataFrame(columns=['mutation_prob', 'crossover_prob', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time'])

# ngens = 100

# mutpb = np.arange(0.05, 0.95, 0.05)

# cxpb = np. arange(0.9, 0.00, -0.05)

# for m in range(0, len(mutpb)):

#     for c in range(0, len(cxpb)):

#         summins = 0

#         sumevals = 0

#         timeofrnds = 0

#         successes = 0

#         sumgens = 0

#         sumsucgenmin = 0

#         sumsucgenevals = 0

#         for rnd in range (0,5):

#             start = time.time()

#             pop, log, hof = run_main ('blend', 'low', 'polynomial', 'high', '3', 'simple', cxpb, mutpb, ngens)

#             endtime = time.time()

#             flag = False

#             for i in range (0, ngens + 1):  #till number of gens + 1

#                 sumevals += (log[i])['nevals']

#                 if ((log[i]['min'] < goal + delta) & (flag == False)):

#                     sumgens += log[i]['gen']

#                     flag = True

#                     sumsucgenmin += log[i]['min']

#                     sumsucgenevals += sumevals

#             summins += hof[0].fitness.values[0]

#             if  hof[0].fitness.values[0] < goal + delta:

#                 successes += 1

#             timeofrnds += endtime - start

#         avgevals = sumevals/5

#         avgtime = timeofrnds/5

#         avgmins = summins/5

#         avgsucgens = sumgens/5

#         avgsumsucgenmin = sumsucgenmin/5

#         avgsumsucgenevals = sumsucgenevals/5

#         gsdf = df.append({'mutation_prob':mutpb[m], 'crossover_prob':cxpb[c], 'successes':successes,'s.avg.min':avgsumsucgenmin, 's.avg.evals':avgsumsucgenevals, 's.avg.gens':avgsucgens, 'avg.evals':avgevals, 'avg.min':avgmins, 'avg.time':avgtime},ignore_index=True)

# display(gsdf)
# import numpy as np

# import random

# from math import sin, cos, sqrt 



# numVariables = 2 



# creator.create( "FitnessMin", base.Fitness , weights=(-1.0,))

# creator.create( "IndividualContainer", list , fitness= creator.FitnessMin)

# toolbox = base.Toolbox()

# toolbox.register( "InitialValue", np.random.uniform, -500, 500)

# toolbox.register( "indiv", tools.initRepeat, creator.IndividualContainer, toolbox.InitialValue, numVariables)

# toolbox.register( "population", tools.initRepeat, list , toolbox.indiv)



# def eval_Rana_Func( indiv ):   #102  

#     t1 = sqrt(abs(indiv[1] + indiv[0] + 1))

#     t2 = sqrt(abs(indiv[1] - indiv[0] + 1))

#     f = (indiv[1] + 1)*cos(t2)*sin(t1) + indiv[0]*cos(t1)*sin(t2)

#     return(f,)  







# from mpl_toolkits.mplot3d import Axes3D

# import matplotlib.pyplot as plt

# from matplotlib import cm

# from matplotlib.ticker import LinearLocator, FormatStrFormatter

# import numpy as np



# X1 = np.arange(-500, 500, 1)

# X2 = np.arange(-500, 500, 1)

# X1, X2 = np.meshgrid(X1, X2)



# Z = np.zeros((len(X1), len(X2)))

# for i in range(0, len(X1)):

#     for j in range(0, len(X2)):

#         indiv = [X1[i][j], X2[i][j]]

#         z = eval_Rana_Func( indiv )

#         Z[i][j] = z[0]



# fig = plt.figure(figsize=(14,8))

# ax = fig.gca(projection='3d')

# surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,

#                        linewidth=0, antialiased=False)

# ax.set_xlabel('x1')

# ax.set_ylabel('x2')

# ax.set_zlabel('f(x1,x2)')

# ax.set_zlim(-600, 600)



# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()
# #penalty



# MIN_BOUND = np.array([-500]*numVariables)

# MAX_BOUND = np.array([500]*numVariables)



# def feasible2( indiv ):

#     if any( indiv < MIN_BOUND) or any( indiv > MAX_BOUND):

#         return False

#     return True



# def distance2( indiv ) :

#     dist = 0.0

#     for i in range (len( indiv )) :

#         penalty = 0

#         if ( indiv [i] < MIN_BOUND[i]) : penalty = -500 - indiv [i]

#         if ( indiv [i] > MAX_BOUND[i]) : penalty = indiv [i] - 500

#         dist = dist + penalty

#     return dist
toolbox.register( "evaluate", eval_Rana_Func)

toolbox.decorate( "evaluate", tools.DeltaPenalty (feasible2, 600.0, distance2))
# eval_Rana_Func([-500, -500])
# #mutation algorithms

# def mut_uniform_high2(individual):

#     individual = mutUniformFloat(individual, -500, 500, 0.5)

#     return individual



# def mut_uniform_low2(individual):

#     individual = mutUniformFloat(individual, -500, 500, 0.05)

#     return individual
# def run_main2 (crossover, crosparams, mutation, mutparams, selectsize, genalg, cxpb, mutpb, ngen):

#     #crossover

#     if (crossover == 'blend'):

#         if (crosparams == 'high'):

#             toolbox.register( "mate", my_cxblend_high)

#         elif (crosparams == 'low'):

#             toolbox.register( "mate", my_cxblend_low)

#     elif (crossover == 'simulated'):

#         if (crosparams == 'high'):

#             toolbox.register( "mate", my_cxsimulated_high)

#         elif (crosparams == 'low'):

#             toolbox.register( "mate", my_cxsimulated_low)

#     #mutation

#     if (mutation == 'gaussian'):

#         if (mutparams == 'high'):

#             toolbox.register( "mutate", tools.mutGaussian, mu = 0, sigma=2.0, indpb=0.5)

#         elif (mutparams == 'low'):

#             toolbox.register( "mutate", tools.mutGaussian, mu = 0, sigma=1.0, indpb=0.05)

#     elif (mutation == 'uniform'):

#         if (mutparams == 'low'):

#             toolbox.register( "mutate", mut_uniform_low2)

#         elif (mutparams == 'high'):

#             toolbox.register( "mutate", mut_uniform_high2)

#     #selection

#     if (selectsize == '3'):

#         toolbox.register( "select", tools.selTournament, tournsize=3)

#     else:

#         toolbox.register( "select", tools.selTournament, tournsize=7)

#     if (genalg == 'simple'):

#         pop, log, hof = ea2_with_stats(cxpb, mutpb,ngen)

#         print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

#         return pop, log, hof

#     elif (genalg == 'mplusl'):

#         pop, log, hof = m_plus_l_with_stats(cxpb, mutpb,ngen)

#         print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

#         return pop, log, hof

#     elif (genalg == 'mcommal'):

#         pop, log, hof = m_comma_l_with_stats(cxpb, mutpb,ngen)

#         print("Best individual is: %s\nwith fitness: %s" % (hof[0], hof[0].fitness))

#         return pop, log, hof



    
# df2 = pd.DataFrame(columns=['operators', 'strategy', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time'])
# #FOR ONE TIME RUNS ONLY



# import time

# ngens = 100

# cxpb = 0.8

# mutpb = 0.2

# goal = -500

# delta = 30

# crossover = 'simulated'

# mutation = 'uniform'

# crosparams = 'high'

# mutparams = 'low'

# selparam = '7'

# ga = 'simple'

# operator = crossover + crosparams + mutation + mutparams + "Sel" + selparam

# summins = 0

# sumevals = 0

# timeofrnds = 0

# successes = 0

# sumgens = 0

# sumsucgenmin = 0

# sumsucgenevals = 0

# for rnd in range (0,5):

#     start = time.time()

#     pop, log, hof = run_main2 (crossover, crosparams, mutation, mutparams, selparam, ga, cxpb, mutpb, ngens)

#     endtime = time.time()

#     flag = False

#     for i in range (0, ngens + 1):  #till number of gens + 1

#         sumevals += (log[i])['nevals']

#         if ((log[i]['min'] < goal + delta) & (flag == False)):

#             sumgens += log[i]['gen']

#             flag = True

#             sumsucgenmin += log[i]['min']

#             sumsucgenevals += sumevals

#     summins += hof[0].fitness.values[0]

#     if  hof[0].fitness.values[0] < goal + delta:

#         successes += 1

#     timeofrnds += endtime - start

# avgevals = sumevals/5

# avgtime = timeofrnds/5

# avgmins = summins/5

# avgsucgens = sumgens/5

# avgsumsucgenmin = sumsucgenmin/5

# avgsumsucgenevals = sumsucgenevals/5

# print("Average number of evaluations:", avgevals)

# print("Average time of each round:" ,avgtime)

# print("Average value of fitness:", avgmins)

# print("Number of successes:", successes)

# if (avgsucgens == 0):

#     print("Average successful generation:", 'none')

# else:

#     print("Average successful generation:", avgsucgens)

# if (sumsucgenmin == 0):

#     print("Average minimum of successful generations:", 'none')

# else:

#     print("Average minimum of successful generations:", avgsumsucgenmin)

# if (sumsucgenevals == 0):

#     print("Average number of evaluations until successful generation:", 'none')

# else:

#     print("Average number of evaluations until successful generation:", avgsumsucgenevals)
# #function run_main (crossover, crosparams, mutation, mutparams, selectsize, genalg)

# #crossover possible values: 'blend', 'simulated'

# #mutation possible values: 'gaussian', 'polynomial'

# #crosparams-mutparams possible values: 'low', 'high'

# #genalg possible values: 'simple', 'mplusl', 'mcommal'

# import time

# ngens = 100

# cxpb = 0.8

# mutpb = 0.2

# goal = -500

# delta = 20

# crossover = ['blend', 'simulated']

# mutation = ['gaussian', 'uniform']

# crosparams = ['low', 'high']

# mutparams = ['low','high']

# selparam = ['3', '7']

# ga = ['simple', 'mplusl', 'mcommal']

# for c in range (0,len(crossover)):

#     for cp in range (0,len(crosparams)):

#         for m in range (0,len(mutation)):

#             for mp in range (0, len(mutparams)):

#                 for s in range (0, len(selparam)):

#                     for al in range (0, len(ga)):

#                         operator = crossover[c] + crosparams[cp] + mutation[m] + mutparams[mp] + "Sel" + selparam[s]

#                         summins = 0

#                         sumevals = 0

#                         timeofrnds = 0

#                         successes = 0

#                         sumgens = 0

#                         sumsucgenmin = 0

#                         sumsucgenevals = 0

#                         for rnd in range (0,5):

#                             start = time.time()

#                             pop, log, hof = run_main2(crossover[c], crosparams[cp], mutation[m], mutparams[mp], selparam[s], ga[al], cxpb, mutpb, ngens)

#                             endtime = time.time()

#                             flag = False

#                             for i in range (0, ngens + 1):  #till number of gens + 1

#                                 sumevals += (log[i])['nevals']

#                                 if ((log[i]['min'] < goal + delta) & (flag == False)):

#                                     sumgens += log[i]['gen']

#                                     flag = True

#                                     sumsucgenmin += log[i]['min']

#                                     sumsucgenevals += sumevals

#                             summins += hof[0].fitness.values[0]

#                             if  hof[0].fitness.values[0] < goal + delta:

#                                 successes += 1

#                             timeofrnds += endtime - start

#                         avgevals = sumevals/5

#                         avgtime = timeofrnds/5

#                         avgmins = summins/5

#                         avgsucgens = sumgens/5

#                         avgsumsucgenmin = sumsucgenmin/5

#                         avgsumsucgenevals = sumsucgenevals/5

#                         df2 = df2.append({'operators':operator, 'strategy':ga[al] , 'successes':successes,'s.avg.min':avgsumsucgenmin, 's.avg.evals':avgsumsucgenevals, 's.avg.gens':avgsucgens, 'avg.evals':avgevals, 'avg.min':avgmins, 'avg.time':avgtime},ignore_index=True)
# display(df2)
pd.set_option('display.max_rows', None)
# #GRIDSEARCH

# gsdf2 = pd.DataFrame(columns=['mutation_prob', 'crossover_prob', 'successes', 's.avg.min', 's.avg.evals', 's.avg.gens', 'avg.evals', 'avg.min', 'avg.time'])

# mutpb = np.arange(0.05, 0.95, 0.05)

# cxpb = np. arange(0.9, 0.00, -0.05)

# ngens = 100

# for m in range(0, len(mutpb)):

#     for c in range(0, len(cxpb)):

#         summins = 0

#         sumevals = 0

#         timeofrnds = 0

#         successes = 0

#         sumgens = 0

#         sumsucgenmin = 0

#         sumsucgenevals = 0

#         for rnd in range (0,5):

#             start = time.time()

#             pop, log, hof = run_main2 ('simulated', 'high', 'gaussian', 'low', '7', 'simple', cxpb[c], mutpb[m], ngens)   #THE BEST

#             endtime = time.time()

#             flag = False

#             for i in range (0, ngens + 1):  #till number of gens + 1

#                 sumevals += (log[i])['nevals']

#                 if ((log[i]['min'] < goal + delta) & (flag == False)):

#                     sumgens += log[i]['gen']

#                     flag = True

#                     sumsucgenmin += log[i]['min']

#                     sumsucgenevals += sumevals

#             summins += hof[0].fitness.values[0]

#             if  hof[0].fitness.values[0] < goal + delta:

#                 successes += 1

#             timeofrnds += endtime - start

#         avgevals = sumevals/5

#         avgtime = timeofrnds/5

#         avgmins = summins/5

#         avgsucgens = sumgens/5

#         avgsumsucgenmin = sumsucgenmin/5

#         avgsumsucgenevals = sumsucgenevals/5

#         gsdf2 = gsdf2.append({'mutation_prob':mutpb[m], 'crossover_prob':cxpb[c], 'successes':successes,'s.avg.min':avgsumsucgenmin, 's.avg.evals':avgsumsucgenevals, 's.avg.gens':avgsucgens, 'avg.evals':avgevals, 'avg.min':avgmins, 'avg.time':avgtime},ignore_index=True)

# display(gsdf2)
# print("The best combination of mutation and crossover propability is in row", gsdf2['avg.min'].idxmin() ) #mtpb = 0.15, cxpb = 0.65
# # Dim > 2     #min(Kd)=(Kd-1) * min(2d) for K = 3,4,... and x_i = -500



# import numpy as np

# import random



# from math import sin, cos, sqrt 



# numVariables =  40         



# creator.create( "FitnessMin", base.Fitness , weights=(-1.0,))

# creator.create( "IndividualContainer", list , fitness= creator.FitnessMin)

# toolbox = base.Toolbox()

# toolbox.register( "InitialValue", np.random.uniform, -500, 500)

# toolbox.register( "indiv", tools.initRepeat, creator.IndividualContainer, toolbox.InitialValue, numVariables)

# toolbox.register( "population", tools.initRepeat, list , toolbox.indiv)



# def eval_Rana_Func_MultiDim( indiv ):    

#     f = 0;

#     for i in range(1, numVariables):

#         t1 = sqrt(abs(indiv[i] + indiv[i-1] + 1))

#         t2 = sqrt(abs(indiv[i] - indiv[i-1] + 1))

#         f += (indiv[i] + 1)*cos(t2)*sin(t1) + indiv[i-1]*cos(t1)*sin(t2)

#     return(f,)  
# ind = toolbox.indiv()

# print(ind)
# indiv=[-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500,-500]  

# f=eval_Rana_Func_MultiDim(indiv)

# print(f)
# #penalty



# MIN_BOUND = np.array([-500]*numVariables)

# MAX_BOUND = np.array([500]*numVariables)



# def feasible( indiv ):

#     if any( indiv < MIN_BOUND) or any( indiv > MAX_BOUND):

#         return False

#     return True



# def distance( indiv ) :

#     dist = 0.0

#     for i in range (len( indiv )) :

#         penalty = 0

#         if ( indiv [i] < MIN_BOUND[i]) : penalty = -500 - indiv [i]

#         if ( indiv [i] > MAX_BOUND[i]) : penalty = indiv [i] - 500

#         dist = dist + penalty

#     return dist
# toolbox.register( "evaluate", eval_Rana_Func_MultiDim)

# toolbox.decorate( "evaluate", tools.DeltaPenality (feasible, 600.0, distance))
# df3 = pd.DataFrame(columns=['D', 'successes', 'avg.min', 'avg.evals', 'avg.time'])
# import time



# crossover = 'simulated'   #best

# crossparams = 'high'

# mutation = 'gaussian'

# mutparams = 'low' 

# selparam = '7'

# genalg = 'simple'

# cxpb = 0.65

# mtpb = 0.15



# operator = crossover + crossparams + mutation + mutparams + "Sel" + selparam



# rounds = 10

# ngen = 100

# goal = -18106

# delta = 2000



# min_fit = 0

# n_evals_total = 0

# timeofrnds = 0

# successes = 0



# for r in range(1, rounds + 1):

#     start = time.time()

#     pop, log, hof = run_main2 ('simulated', 'high', 'gaussian', 'low', '7', 'simple', cxpb, mtpb, ngen)

#     endtime = time.time()

#     timeofrnds += (endtime - start)

#     min_fit += hof[0].fitness.values[0]

#     if  hof[0].fitness.values[0] < goal + delta:

#         successes += 1

#     for g in range(0, ngen + 1):

#         n_evals_total += (log[g])['nevals']

    

# avg_min = min_fit/rounds    

# avg_evals = n_evals_total/rounds

# avg_time = timeofrnds/rounds



# df3 = df3.append({'D':numVariables, 'successes':successes, 'avg.evals':avg_evals, 'avg.min':avg_min, 'avg.time':avg_time},ignore_index=True)
# display(df3)