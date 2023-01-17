import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import random

import math

import queue 

import os

import time

from numpy.linalg import norm

from collections import Counter

from matplotlib import collections as mc
df1 = pd.read_csv('../input/cp2410-a2-cities/cities.csv')

df1.head()
def isPrime(n):

    primes = [True for i in range(n+1)] 

    primes[0] = False 

    primes[1] = False

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)
nb_cities = max(df1.CityId)

print ("Number of cities to visit: ", nb_cities)
nb_cities = max(df1.CityId)

primes = np.array(isPrime(nb_cities)).astype(int)

df1['P'] = isPrime(max(df1.CityId))

df2 = isPrime(max(df1.CityId))

df1.head(10)
def calcTotalDist(arr):

        total_distance = 0



        for i in range(0, len(arr) - 1):

                first_point = data_dict[arr[i]]

                second_point = data_dict[arr[i + 1]]



                total_distance += math.sqrt(pow((second_point[0] - first_point[0]), 2) + pow((second_point[1] - first_point[1]), 2))

        return total_distance
data = pd.read_csv("cities.csv")

data_use = 1



# Remove unwanted rows of data

data_cutoff = int(data.count(0)['X']*data_use)

data = data.drop(data.index[data_cutoff:])

origin = data[data.CityId == 0]



data['Distance'] = np.sqrt(pow((data['X'] - float(origin.X)), 2) + pow((data['Y'] - float(origin.Y)), 2))





# Put data into dictionary

data_dict = {}

index_list = []





for index, row in data.iterrows():

        data_dict[row['CityId'].astype(int)] = (row['X'].astype(float), row['Y'].astype(float), row['Distance'].astype(float))

        index_list.append(row['CityId'].astype(int))





def mergeSort(arr):

    t1 = time.time()

    time_array.append(t1 - start_time)



    if len(arr) > 1:

        mid = len(arr) // 2

        left_split = arr[:mid]

        right_split = arr[mid:]



        mergeSort(left_split)

        mergeSort(right_split)



        i = j = k = 0



        # Copy data to temp arrays left_split[] and right_split[]

        while i < len(left_split) and j < len(right_split):



            if data_dict[left_split[i]][2] < data_dict[right_split[j]][2]:

                arr[k] = left_split[i]

                i += 1

            else:

                arr[k] = right_split[j]

                j += 1

            k += 1



        # Checking if any element was left

        while i < len(left_split):

            arr[k] = left_split[i]

            i += 1

            k += 1



        while j < len(right_split):

            arr[k] = right_split[j]

            j += 1

            k += 1

            



# Run Function

time_array = []



start_dist = calcTotalDist(index_list)

print("Start Distance: ", start_dist)



start_time = time.time()

mergeSort(index_list)

finish_time = time.time()



total_time = finish_time-start_time

time_array.append(total_time)



sorted_dist = calcTotalDist(index_list)

print("Sorted Distance: " + "{:,}".format(sorted_dist))

print("Improvement: ", 100-(sorted_dist/start_dist)*100, "%")

print("Total Time: ", finish_time-start_time, "s")



# Time vs Recursion Count Graph

bars = range(1, len(time_array) + 1)

y_pos = np.arange(len(bars))

plt.plot(y_pos, time_array)



plt.title('Merge Sort Time Graph')

plt.xlabel('No. of Iterations')

plt.ylabel('Time (Sec)')



plt.show()
data = pd.read_csv("cities.csv")

data_use = 1



# Remove unwanted rows of data

data_cutoff = int(data.count(0)['X']*data_use)

data = data.drop(data.index[data_cutoff:])

origin = data[data.CityId == 0]



data['Distance'] = np.sqrt(pow((data['X'] - float(origin.X)), 2) + pow((data['Y'] - float(origin.Y)), 2))





# Put data into dictionary

data_dict = {}

index_list = []





for index, row in data.iterrows():

        data_dict[row['CityId'].astype(int)] = (row['X'].astype(float), row['Y'].astype(float), row['Distance'].astype(float))

        index_list.append(row['CityId'].astype(int))

        

        

def insertionSort(arr):



    # Iterate through the array

    for i in range(1, len(arr)):



        t1 = time.time()

        time_array.append(t1 - start_time)



        key = arr[i]



        j = i - 1

        while j >= 0 and data_dict[key][2] < data_dict[arr[j]][2]:

            arr[j + 1] = arr[j]

            j -= 1

        arr[j + 1] = key





# Run Function

time_array = []



start_dist = calcTotalDist(index_list)

print("Start Distance: ", start_dist)



start_time = time.time()

insertionSort(index_list)

finish_time = time.time()



sorted_dist = calcTotalDist(index_list)

print("Sorted Distance: " + "{:,}".format(sorted_dist))

print("Improvement: ", 100-(sorted_dist/start_dist)*100, "%")

print("Total time: ", finish_time - start_time, "s")





# Time vs Recursion Count Graph

bars = range(1, len(time_array) + 1)

y_pos = np.arange(len(bars))

plt.plot(y_pos, time_array)



plt.title('Insertion Sort Time Graph')

plt.xlabel('No. of Iterations')

plt.ylabel('Time (Sec)')



plt.show()
def calDistance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))



def findMin(listEdge, visited):

    if len(visited) == 0:# if the list of visited city is empty -> find the minimun

        return min(listEdge, key= listEdge.get) ## Find the minimun edge value of vertex

    else:

        # if the list of visited city is not empty

        for key in visited:

            listEdge.pop(key)

        return min(listEdge, key= listEdge.get) ## Find the minimum edge value of vertex
data = pd.read_csv('cities.csv')

# Any results you write to the current directory are saved as output.



cities = []

for i in range(len(data)):

    edge={} # create a dictionary to contain all the edge which contain city as a key and distance as value

    for j in range(len(data)):

        edge[j] = calDistance(data.values[i][1],data.values[i][2],data.values[j][1],data.values[j][2])

    cities.append(edge)
visited = []# which will be contain the visited city and also for the path

cost = []# contain all the distance every step

position = 0

visited.append(position)



while len(visited) < len(cities):

    tempt = position # assign current position into a tempt variavle which will use for find the distance

    position = findMin(cities[position], visited)# find the city which is near to current city

    cost.append(cities[tempt][position])# add the distance into cost list

    visited.append(position)# add the visited city into list

# at the end add the zero city into the visited list to complete the path 

# and also calculate the distance from the last city to zero city and add it into cost list



visited.append(0)

cost.append(calDistance(data.values[0][1],data.values[0][2],data.values[position][1],data.values[position][2]))    
distance = 0 

step = 0 

flag = False # which will let you know whether we met the end because there are 2 zero element in the visited list

for city in visited: # go through all the city in the list

    if city == 0 and flag == False: #start with the 0 city

        distance = distance + cost[step]*1.1

        step += 1

        flag = True # just for separate between 0 at beginning and 0 at the end

    elif city == 0 and flag == True: #end at the city 0

        break;

    elif step % 10 == 0 and isPrime(city) == False:#if there is a 10th step and not a prime city 

        distance = distance + cost[step]*1.1

        step += 1

    else:

        distance = distance + cost[step]

        step += 1    

        

print("Distance calculated using Map: " + "{:,}".format(distance))
final_path = pd.DataFrame(visited,columns=["idCity"])

final_path.to_csv("./final_path.csv")
import pandas as pd

cities = pd.read_csv("../input/cities.csv")