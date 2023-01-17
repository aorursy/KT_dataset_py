# A system that uses a genetic algorithm to maximize a function of many variables

import random

import sys

import math

from math import *

from decimal import Decimal, localcontext



# Simpler fitness_function of two variables with a maximum at (x=1, y=2)

def simple_fitness_function(x, y):

    return - (x**2) + (2 * x) - (y ** 2) + (4 * y)

#     return  -x**2-y**2 - 2*x*y
# Takes a function and list of arguments, applies function to arguments

def evaluate_generation(population):

    scores = []

    total = 0

    for individual in population:

        r = simple_fitness_function(individual[0], individual[1])

        scores.append(r)

        total += r

        

    avg = total / len(scores)

    return scores, avg
# Create child from parent

def mutate(individual):

    new = []

    for attribute in individual:

        new.append(attribute + random.normalvariate(0, attribute + .1))  # Random factor of normal distribution

    return new
# Given a population, return the best individual and the associated value

def find_best(population):

    best = None

    val = None

    for individual in population:

        if len(individual) == 2:

            r = simple_fitness_function(individual[0], individual[1])

            try:

                if r > val:

                    best = individual

                    val = r

            except:  # On the first run, set the result as best

                    best = individual

                    val = r

    return best, val
# Create a population of p lists of [0, 0, ..., 0] of length n

def initialize(n, p):

    pop = [[0] * n]

    for i in range(p):

        pop.append(mutate(pop[0]))

    return pop
# Handle the output of the genetic algorithm

def termination(best, val, total_iterations, population_size, num_attributes):

    best = [round(x, 3) for x in best]  #  Round for printing

    print("Ran", total_iterations, "iterations on a population of", population_size)

    print("The optimal input is", best, "with a value of", round(val, 3))
num_attributes = 2  ###  z = f(x,y)

population_size = 1000

total_iterations = 1000



population = initialize(num_attributes, population_size)

for iteration in range(total_iterations):

    scores, avg = evaluate_generation(population)

    

    deleted = 0

    ### selection

    new_population = []

    for i in range(len(population)):

        if scores[i] < avg:

            deleted += 1

        else:

            new_population.append(population[i])

    

    #mutation

    for i in range(deleted):

        new_population.append(mutate(new_population[i % len(new_population)]))

    

    population = new_population

best, val = find_best(population)

termination(best, val, total_iterations, population_size, num_attributes)
# plot 

from mpl_toolkits import mplot3d

import numpy as np

import matplotlib.pyplot as plt

def f(x, y):

    return -x**2+2*x-y**2+4*y

#     return -x**2-y**2 - 2*x*y



# f2 = np.vectorize(f)



x = np.linspace(-100, 100, 100)

y = np.linspace(-100, 100, 100)

X, Y = np.meshgrid(x, y)

Z = f(X, Y)

fig = plt.figure()

ax = plt.axes(projection='3d')

ax.contour3D(X, Y, Z, 100, cmap='binary')

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')

fig
ax.scatter3D(best[0], best[1], val, c='red', cmap='viridis')

ax.view_init(60, 20)

fig
ax = plt.axes(projection='3d')

ax.plot_surface(X, Y, Z, rstride=1, cstride=1,

                cmap='viridis', edgecolor='none')

ax.set_title('surface')

# ax.scatter3D(best[0], best[1], val, c='red', cmap='viridis');

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_zlabel('z')