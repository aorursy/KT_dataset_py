# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from scipy.stats import wilcoxon



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import math  

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
LB=-1

UB=1

Npop=40

NumOfRivers=5

NumOfStreams=Npop-NumOfRivers-1

sea_cost=0

dmax=0.5

n_iterations=500

D=1000

c1 = 0.5

target = D*1.0

target_error = 0.1

def sigmoid(x):

  return 1 / (1 +np.exp(-x))

def fitness_function(position_vector):

    toplam=0

    for i in position_vector:

        if sigmoid(i)>=0.5:

            toplam=toplam+1

    return toplam
position_vector = np.array([np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()for _ in range(D)])

                            for _ in range(Npop)])

print(position_vector.shape)
cost_vector=np.array([0 for _ in range(Npop)])

cost_vector_river=np.array([0 for _ in range(NumOfRivers)])

for i in range(Npop):

    cost_vector[i]=fitness_function(position_vector[i])

#print(cost_vector)

max_indice=cost_vector.argmax(axis=0)

sea_position=position_vector[max_indice].copy()

sea_cost= fitness_function(sea_position)

print("Sea Cost: ",sea_cost)

position_vector[max_indice]=position_vector[0]

position_vector[0]=sea_position

river_positions=position_vector[1:(NumOfRivers+1)]

stream_positions=position_vector[(NumOfRivers+1):]

print("Sea Position Vector Shape: ",sea_position.shape)

print("River Positions Vector Shape: ",river_positions.shape)

print("Streams Position Vector Shape: ",stream_positions.shape)
def rivers_to_sea(sea_cost,sea_position):

    for i in range(NumOfRivers):

        cost_vector[i]=fitness_function(river_positions[i])

        if cost_vector[i]>sea_cost:

            new_sea=river_positions[i].copy()

            river_positions[i]=sea_position.copy()

            sea_position=new_sea

        else:

            new_position=river_positions[i]+ c1*random.random() * (sea_position - river_positions[i])

            for d in range(new_position.shape[0]):

                if new_position[d]> UB:

                    new_position[d]=UB

                if new_position[d]<LB:

                    new_position[d]=LB

            river_positions[i] = new_position.copy()

    return sea_position



def streams_to_sea(sea_cost,sea_position):

    for i in range(NumOfStreams):

        cost_vector[i]=fitness_function(stream_positions[i])

        if cost_vector[i]>sea_cost:

            new_sea=stream_positions[i].copy()

            stream_positions[i]=sea_position.copy()

            sea_position=new_sea

        else:

            new_position=stream_positions[i]+ c1*random.random() * (sea_position - stream_positions[i])

            for d in range(new_position.shape[0]):

                if new_position[d]> UB:

                    new_position[d]=UB

                if new_position[d]<LB:

                    new_position[d]=LB

            stream_positions[i] = new_position.copy()

    return sea_position

def streams_to_rivers(sea_cost,sea_position):

    for i in range(NumOfStreams):

        cost_vector[i]=fitness_function(stream_positions[i])

        if cost_vector[i]>sea_cost:

            new_sea=stream_positions[i].copy()

            stream_positions[i]=sea_position.copy()

            sea_position=new_sea

        else :

            for j in range(NumOfRivers):

                cost_vector_river[j]=fitness_function(river_positions[j])

                if cost_vector_river[j]<cost_vector[i]:

                    new_river=stream_positions[i].copy()

                    stream_positions[i]=river_positions[j].copy()

                    river_positions[j]=new_river

                else:

                    new_position=river_positions[j]+ c1*random.random() * (sea_position - river_positions[j])

                    for d in range(new_position.shape[0]):

                        if new_position[d]> UB:

                            new_position[d]=UB

                        if new_position[d]<LB:

                            new_position[d]=LB

                    river_positions[j] = new_position.copy()

    return sea_position
def evaporation(sea_position):

    for i in range(NumOfStreams):

        if fitness_function(sea_position-stream_positions[i])>dmax:

            new_position=stream_positions[i]*np.zeros(D)+LB+(2*random.random()*(sea_position-stream_positions[i]))

            for d in range(new_position.shape[0]):

                if new_position[d]> UB:

                    new_position[d]=UB

                if new_position[d]<LB:

                    new_position[d]=LB

            stream_positions[i] = new_position

    for i in range(NumOfRivers):

        if fitness_function(sea_position-river_positions[i])>dmax:

            new_position=river_positions[i]*np.zeros(D)+LB+(2*random.random()*(sea_position-river_positions[i]))

            for d in range(new_position.shape[0]):

                if new_position[d]> UB:

                    new_position[d]=UB

                if new_position[d]<LB:

                    new_position[d]=LB

            river_positions[i] = new_position

iteration = 0

while iteration < n_iterations or iteration == n_iterations:

    sea_position=streams_to_rivers(sea_cost,sea_position)

    sea_cost= fitness_function(sea_position)

    sea_position=rivers_to_sea(sea_cost,sea_position)

    sea_cost= fitness_function(sea_position)

    sea_position=streams_to_sea(sea_cost,sea_position)

    sea_cost= fitness_function(sea_position)

    evaporation(sea_position)

    sea_cost= fitness_function(sea_position)

    print("The best cost value",sea_cost, "in iteration number ", iteration)   

    if(target-sea_cost < target_error):

        break

    iteration = iteration + 1

    dmax=dmax-(dmax/n_iterations)

print(sea_position)