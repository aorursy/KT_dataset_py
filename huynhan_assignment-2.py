import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

import math

import random

def isPrime(number):

    if number > 1:

        for i in range(2, number):

            if number % i == 0 and i != 1:

                return False

                break

        else:

            return True

    else:

        return False
def calDistance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2))
def quickSort(alist):

    quickSortHelper(alist,0,len(alist)-1)



def quickSortHelper(alist,first,last):

    if first<last:

        splitpoint = partition(alist,first,last)

        quickSortHelper(alist,first,splitpoint-1)

        quickSortHelper(alist,splitpoint+1,last)



def partition(alist,first,last):

    pivotvalue = alist[first]

    leftmark = first+1

    rightmark = last

    done = False

    while not done:

        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:

            leftmark = leftmark + 1

        while alist[rightmark] >= pivotvalue and rightmark >= leftmark:

            rightmark = rightmark -1



        if rightmark < leftmark:

            done = True

        else:

            temp = alist[leftmark]

            alist[leftmark] = alist[rightmark]

            alist[rightmark] = temp

    temp = alist[first]

    alist[first] = alist[rightmark]

    alist[rightmark] = temp

    return rightmark
def findMin(listEdge, visited):

    if len(visited) == 0:# if the list of visited city is empty -> find the minimun

        return min(listEdge, key= listEdge.get) ## Find the minimun edge value of vertex

    else:

        # if the list of visited city is not empty

        for key in visited:

            listEdge.pop(key)

        return min(listEdge, key= listEdge.get) ## Find the minimun edge value of vertex

            
cities = pd.read_csv('../input/cities.csv')

small_sample = cities[0:500]



idCity = [small_sample.values[i][0] for i in range(len(small_sample))]

x_coor = [float(small_sample.values[i][1]) for i in range(len(small_sample))]

y_coor = [float(small_sample.values[i][2]) for i in range(len(small_sample))]



x_sorted = [small_sample.values[i][1] for i in range(1,len(small_sample))]# take out sample without firts city

quickSort(x_sorted)



xCoordinateSorted_result = []

def distance_cost():

    distance = 0

    step = 0

    for i in range(0,(len(x_sorted)-1)):  

        position = x_coor.index(x_sorted[i])

        if i == 0:  #start with 0 city first

            tmp = calDistance(x_coor[0], x_sorted[i], y_coor[0], y_coor[position])

            distance = distance + tmp* 1.1

            step += 1

            xCoordinateSorted_result.append(0)

        elif i == len(x_sorted): #if travel to the last city then travel back to the 0 city

            distance = calDistance(x_sorted[i], x_coor[0], y_coor[position], y_coor[0])

            distance = distance + tmp

            xCoordinateSorted_result.append(x_coor.index(x_sorted[i]))

            xCoordinateSorted_result.append(0)

            

        elif step % 10 == 0 and isPrime(position) == False: #if there is a 10th step and not a prime city 

            tmp = calDistance(x_sorted[i], x_sorted[i + 1], y_coor[position], y_coor[x_coor.index(x_sorted[i+1])])

            distance = distance + tmp* 1.1  #then distance will be increase 10%

            step += 1

            xCoordinateSorted_result.append(x_coor.index(x_sorted[i]))

            

        else:

            tmp = calDistance(x_sorted[i], x_sorted[i + 1], y_coor[position], y_coor[x_coor.index(x_sorted[i+1])])

            distance += tmp

            step += 1

            xCoordinateSorted_result.append(x_coor.index(x_sorted[i]))

    return distance

distance = distance_cost()
print("using dumbest way to calculate the distance with penalized distance after sorting base on x, distance = " + str(distance))
# print(os.listdir("../input/cities.csv"))

data = pd.read_csv('../input/cities.csv')

sample = data[0:500]

# Any results you write to the current directory are saved as output.
cities = []

for i in range(len(sample)):

    edge={} # create a dictionary to contain all the edge which contain city as a key and distance as value

    for j in range(len(sample)):

        edge[j] = calDistance(sample.values[i][1],sample.values[i][2],sample.values[j][1],sample.values[j][2])

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

cost.append(calDistance(sample.values[0][1],sample.values[0][2],sample.values[position][1],sample.values[position][2]))
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
print("using list of dictionary structure to calculate the distance with penalized distance , distance = " + str(distance))
path_summited = pd.DataFrame(visited,columns=["idCity"])

path_summited.to_csv("./path_summited.csv")
