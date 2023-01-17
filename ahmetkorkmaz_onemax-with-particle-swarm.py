# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random





#function that models the problem

def sigmoid(x):

  return 1 / (1 +np.exp(-x))

def fitness_function(position_vector):

    toplam=0

    for i in position_vector:

        if sigmoid(i)>=0.5:

            toplam=toplam+1

    return toplam

#Some variables to calculate the velocity

W = 0.005

c1 = 0.005

c2 = 0.009

n_particles = 40

D=1000

target = D*1.0

n_iterations = 300

target_error = 0.1

LB=-1

UB=1



position_vector = np.array([np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()for _ in range(D)])for _ in range(n_particles)])

pbest_position=position_vector.copy()

pbest_fitness_value = np.zeros(n_particles)

gbest_position=np.zeros(D)

gbest_fitness_initial = 0.0

for i in range(n_particles):

    fitness_cadidate = fitness_function(position_vector[i])

    pbest_fitness_value[i] = fitness_cadidate

    if(gbest_fitness_initial < fitness_cadidate):

        gbest_fitness_initial = fitness_cadidate

        gbest_position=position_vector[i]



velocity_vector= np.array([random.random()for _ in range(D)])
iteration = 0

while iteration < n_iterations:

    iteration = iteration + 1

    for i in range(n_particles):

        fitness_cadidate = fitness_function(position_vector[i])

        if(pbest_fitness_value[i] < fitness_cadidate):

            pbest_fitness_value[i] = fitness_cadidate

        if(gbest_fitness_initial < fitness_cadidate):

            gbest_fitness_initial = fitness_cadidate

            gbest_position=position_vector[i]

    print("The best positions fitness value is: ",gbest_fitness_initial,"up to iteration number ", iteration)

    if(target-gbest_fitness_initial < target_error):

            break

    for i in range(n_particles):

        a=velocity_vector[i] 

        new_velocity= (W*a) + (c1*random.random()) * (pbest_position[i] - position_vector[i]) + (c2*random.random()) * (gbest_position-position_vector[i])

        new_position = new_velocity + position_vector[i]

        for d in new_position:

            d = np.maximum(d, LB);

            d = np.minimum(d, UB);

        a=new_velocity

        position_vector[i] = new_position.copy()

print("The best positions are ",gbest_position, "in iteration number ", iteration)