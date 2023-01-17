# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import matplotlib.pyplot as plt

from math import exp



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
input_data = []

class_labels = []



#reading data in file

filename = '/kaggle/input/iris.txt'

file = open(filename,'r')



for line in file.readlines():

    row = line.strip().split(',')

    input_data.append(row[0:4])

    class_labels.append(row[4])

file.close



input_data = np.asarray(input_data,dtype=np.float32)



print("=======================================")

print("Data Loaded from ",filename)

print("=======================================")



input_data = np.asarray(input_data,dtype=np.float32)

print("=======================================")

print(input_data)

print("=======================================")
input_dimensions = 4

som_width = 8

som_height = 8



coordinate_map = np.zeros([som_height,som_width,2],dtype=np.int32)

for i in range(0,som_height):

    for j in range(0,som_width):

          coordinate_map[i][j] = [i,j]



SOM = np.random.uniform(size=(som_height,som_width,input_dimensions))

prev_SOM = np.zeros((som_height,som_width,input_dimensions))



radius_init_value = max(som_width,som_height)/2

radius = radius_init_value

epochs = 500

learning_rate_init_value = 0.1

learning_rate = learning_rate_init_value

convergence = [1]

distance_lower_limit=0.0001 

is_converged=False



epoch=0

while epoch < epochs:

    print("=======================================")

    print("Epoch = ", epoch)

    print("=======================================")

    

    shuffle = np.random.randint(len(input_data), size=len(input_data))

    for i in range(len(input_data)):  #for each input

        distance = np.linalg.norm(SOM - prev_SOM)

        if  distance <= distance_lower_limit:

            is_converged=True

            break

        else:

            pattern = input_data[shuffle[i]]

            pattern_array = np.tile(pattern, (som_height, som_width, 1))#create array using the pattern



            distance_map = np.linalg.norm(pattern_array - SOM, axis=2)#calculate distances  

            flat_index_of_BMU = np.argmin(distance_map, axis=None)#get flat index of BMU

            #print("Flat Index = ", flat_index_of_BMU)

            BMU = np.unravel_index(flat_index_of_BMU, distance_map.shape)#calculate indices of the BMU using flat index

            #print("BMU Indices = ",BMU)



            prev_SOM = np.copy(SOM)

             

            for i in range(som_height):#update weights of the neighbour

                for j in range(som_height):

                    local_distance = np.linalg.norm([i - BMU[0], j - BMU[1]])

                    if local_distance <= radius:

                        SOM[i][j] = SOM[i][j] + learning_rate*(pattern-SOM[i][j])

            

            learning_rate = learning_rate_init_value*(1-(epoch/epochs))

            radius = radius_init_value*math.exp(-epoch/epochs)

            

    if distance < min(convergence):

        print('Lower distance discovered: %s' %str(distance) + ' at epoch: %s' % str(epoch))

        print('\tLearning rate: ' + str(learning_rate) + '\tNeighbourhood radius: ' + str(radius))

        MAP_final = SOM

    convergence.append(distance)

    

    if is_converged==True:

        break

    epoch+=1
#plot graph

plt.plot(convergence)

plt.ylabel('error')

plt.xlabel('epoch')

plt.grid(True)

plt.yscale('log')

plt.show()



print('Final Distance Value: ' + str(distance))



BMU_index = np.zeros([2],dtype=np.int32)

class_map = np.zeros([som_height,som_width,3],dtype=np.float32)



i=0

for pattern in input_data:

    

    pattern_ary = np.tile(pattern, (som_height, som_width, 1))

    distance_map = np.linalg.norm(pattern_ary - MAP_final, axis=2)

    BMU_index = np.unravel_index(np.argmin(distance_map, axis=None), distance_map.shape)

    

    x = BMU_index[0]

    y = BMU_index[1]

    

    if class_labels[i] == '1':

        if class_map[x][y][0] <= 0.5:

            class_map[x][y] += np.asarray([0.5,0,0])

    elif class_labels[i] == '2':

        if class_map[x][y][1] <= 0.5:

            class_map[x][y] += np.asarray([0,0,0.5])

    elif class_labels[i] == '3':

        if class_map[x][y][2] <= 0.5:

            class_map[x][y] += np.asarray([0,0.5,0])

    i+=1

class_map = np.flip(class_map,0)

    

print("Red = Class 1")

print("Green = Class 2")

print("Blue = Class 3")



plt.imshow(class_map, interpolation='nearest')