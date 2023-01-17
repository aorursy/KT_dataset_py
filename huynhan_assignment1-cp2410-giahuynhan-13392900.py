import os

import math

import queue

import pandas as pd

cities = pd.read_csv('../input/cities.csv')

small_sample = cities[0:int(len(cities)*0.1)]



idCity = [small_sample.values[i][0] for i in range(len(small_sample))]

x_coor = [float(small_sample.values[i][1]) for i in range(len(small_sample))]

y_coor = [float(small_sample.values[i][2]) for i in range(len(small_sample))]
def isPrime(number):

    if number > 1:# number must greater than 1 because 0 and 1 are not prime number

        for i in range(2, number):# then check whether this number by devide for other number from 2 to (this number -1)

            if number % i == 0 and i != 1:# if this number is evenly divide for other number instead of 1

                return False #then this is not prime number

                break

        else:# if not this is prime number 

            return True

    else:# if the number is smaller or equal to 1 than this is not prime number. 

        return False
def insertionSort(alist):

    for index in range(1,len(alist)):# run loop from second index to the length of list

        currentvalue = alist[index] #assign value of a list in this index to currentvalue variable

        position = index #assign this index into position 



        #to check the order of list from the head until the current position

        while position>0 and alist[position-1]>currentvalue:

            # swap position if the previous position value is smaller than value of current position

            alist[position]=alist[position-1]

            position = position-1



        alist[position]=currentvalue
# calculate the distance between 2 city

def calDistance(x1, y1, x2, y2):

    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
"""

calculate distance follow order in file without penalties

"""

def distance_cost():

    distance =0

    distance = calDistance(x_coor[0], x_coor[1], y_coor[0], y_coor[1]) #start with 0 city first

    for i in range(1, (len(idCity)-1)):  # then loop to calculate distance for the rest

        if i == len(idCity):# if the last city then calculate distance between it and 0 city

            distance = calDistance(x_coor[i], x_coor[0], y_coor[i], y_coor[0])

            distance = distance + tmp

        else:

            tmp = calDistance(x_coor[i], x_coor[i + 1], y_coor[i], y_coor[i + 1])

            distance = distance + tmp

    return distance

distance = distance_cost()
print("using dumbest way to calculate the distance without penalties, distance = " + str(distance))
"""

calculate distance follow order in file with penalized distance

"""

dumbest_result = []

def distance_cost():

    distance = 0

    step = 0

    for i in range((len(idCity)-1)):# loop to run through all city

        if i == 0:#start with 0 city first

            tmp = calDistance(x_coor[i], x_coor[i + 1], y_coor[i], y_coor[i + 1])

            distance = distance + tmp* 1.1

            step += 1

            dumbest_result.append(i)

        elif i == len(idCity):# if the last city then calculate distance between it and 0 city

            distance = calDistance(x_coor[i], x_coor[0], y_coor[i], y_coor[0])

            distance = distance + tmp

            step += 1

            dumbest_result.append(i)

            dumbest_result.append(0)

        elif step % 10 == 0 and isPrime(i) == False and i!= 0:#if there is a 10th step and not a prime city 

            tmp = calDistance(x_coor[i], x_coor[i + 1], y_coor[i], y_coor[i + 1])

            distance = distance + tmp* 1.1#then distance will be increase 10%

            step += 1

            dumbest_result.append(i)

        else:

            tmp = calDistance(x_coor[i], x_coor[i + 1], y_coor[i], y_coor[i + 1])

            distance += tmp

            step += 1

            dumbest_result.append(i)

    return distance

distance = distance_cost()
print("using dumbest way to calculate the distance with penalized distance, distance = {}".format(distance))
dumbest_submission = pd.DataFrame(dumbest_result,columns=["idCity"])

dumbest_submission.to_csv("./dumbest_submission.csv")
cities = pd.read_csv('../input/cities.csv')

small_sample = cities[0:int(len(cities)*0.1)]

idCity = [small_sample.values[i][0] for i in range(len(small_sample))]

x_coor = [small_sample.values[i][1] for i in range(len(small_sample))]

y_coor = [small_sample.values[i][2] for i in range(len(small_sample))]
x_sorted = [small_sample.values[i][1] for i in range(1,len(small_sample))]# take out sample without firts city

insertionSort(x_sorted) # I will sort base on x
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
xCoordinateSorted_submission = pd.DataFrame(xCoordinateSorted_result,columns=["idCity"])

xCoordinateSorted_submission.to_csv("./xCoordinateSorted_submission.csv")
cities = pd.read_csv('../input/cities.csv')

small_sample = cities[0:int(len(cities)*0.1)]

idCity = [small_sample.values[i][0] for i in range(len(small_sample))]

x_coor = [small_sample.values[i][1] for i in range(len(small_sample))]

y_coor = [small_sample.values[i][2] for i in range(len(small_sample))]
y_sorted = [small_sample.values[i][2] for i in range(1,len(small_sample))]# take out sample without firts city

insertionSort(y_sorted) # I will sort base on x
yCoordinateSorted_result = []

def distance_cost():

    distance = 0

    step = 0

    for i in range(0,(len(y_sorted)-1)):  

        position = y_coor.index(y_sorted[i])

        if i == 0:  #start with 0 city first

            tmp = calDistance(x_coor[0], x_coor[position], y_coor[0], y_sorted[i])

            distance = distance + tmp* 1.1

            step += 1

            yCoordinateSorted_result.append(0)

        elif i == len(x_sorted):# if travel to last city then come back to the 0 city

            distance = calDistance(x_coor[postion], x_coor[0], y_sorted[i], y_coor[0])

            distance = distance + tmp

            yCoordinateSorted_result.append(y_coor.index(y_sorted[i]))

            yCoordinateSorted_result.append(0)

        elif step % 10 == 0 and isPrime(position) == False: #if there is a 10th step and not a prime city 

            tmp = calDistance(x_coor[position], x_coor[y_coor.index(y_sorted[i+1])], y_sorted[i], y_sorted[i + 1])

            distance = distance + tmp* 1.1  #then distance will be increase 10%

            step += 1

            yCoordinateSorted_result.append(y_coor.index(y_sorted[i]))

        else:

            tmp = calDistance(x_coor[position], x_coor[y_coor.index(y_sorted[i+1])], y_sorted[i], y_sorted[i + 1])

            distance += tmp

            step += 1

            yCoordinateSorted_result.append(y_coor.index(y_sorted[i]))

    return distance

distance = distance_cost()
print("using dumbest way to calculate the distance with penalized distance after sorting base on x, distance = " + str(distance))
yCoordinateSorted_submission = pd.DataFrame(yCoordinateSorted_result,columns=["idCity"])

yCoordinateSorted_submission.to_csv("./yCoordinateSorted_submission.csv")