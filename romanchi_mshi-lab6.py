import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import math



DNA_SIZE = 1

DNA_BOUND = [-10, 10]

N_GENERATIONS = 600

POP_SIZE = 100

N_KID = 50



def F_Matias(x, y): 

    return 0.26* (x**2 + y**2) - 0.48*x*y

def F_Levi13(x, y):

    return pow(math.sin(3*math.pi*x),2) + pow(x-1,2)*(1+pow(math.sin(3*math.pi*y),2)) + pow(y-1,2)*(1+ pow(2*math.pi*y,2))

def Shaffer4(x, y):

    nominator = pow(math.cos(math.sin(math.fabs(x**2 - y**2))),2) - 0.5

    dominator = pow(1 + 0.001*(x**2 + y**2), 2)

    return 0.5 + float(nominator)/float(dominator)



def F(point):

    x = point[0]

    y = point[1]

    return Shaffer4(x, y)



def create_kids(pop, n_kid):

    kids = {'DNA': np.zeros((N_KID, 2))}

    kids['mut_strength'] = np.empty_like(kids['DNA'])



    for kid, mut_str in zip(kids['DNA'], kids['mut_strength']):

        parent1, parent2 = np.random.choice(np.arange(POP_SIZE), size=2, replace=False)

        crossover_koeff = np.random.randint(0,1)

        kid[crossover_koeff] = pop['DNA'][parent1][crossover_koeff]

        kid[1 if crossover_koeff == 0 else 0] = pop['DNA'][parent2][1 if crossover_koeff == 0 else 0]

        

        #mutation

        mut_str = mut_str + (np.random.rand(*mut_str.shape)-0.5)

        kid += mut_str * np.random.rand(*mut_str.shape)

        kid[:] = np.clip(kid, *DNA_BOUND)

    return kids

        

def nature_choose(pop,kids):

    for key in ['DNA', 'mut_strength']:

        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = [F([x,y])for x,y in pop['DNA']] # C

    data = pd.DataFrame(pop['DNA'])

    data['fitness'] = fitness

    data = data.sort_values(by=['fitness'],ascending=True)

    data = data.head(100)

    del data['fitness']

    new_pop = {'DNA': data.values}

    new_pop['mut_strength'] = np.empty_like(new_pop['DNA'])

    return new_pop

pop = {'DNA': np.zeros((POP_SIZE, 2)) + 1}

pop['mut_strength'] = np.empty_like(pop['DNA'])



for _ in range(N_GENERATIONS):

    kids = create_kids(pop, N_KID)

    pop = nature_choose(pop, kids)

    

result = pd.DataFrame(pop['DNA'])

result['distance'] = [pow(point[0]**2 + point[1]**2, 0.5) for point in pop['DNA']]

result = result.sort_values(by=['distance'])

print(result)