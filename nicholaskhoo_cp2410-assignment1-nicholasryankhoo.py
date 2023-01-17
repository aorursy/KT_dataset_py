from collections import deque #importing deque to use to store final order of the cities

import csv  # imported to open and read the .csv file

import math # imported to use the math.sqrt function 

import sympy # imported to use sympy.isprime function to check for prime numbers

import time # imported to print time taken to run 
# Open cities.csv file

with open('../input/cities.csv', 'r') as file:

    reader = csv.reader(file)

    cities = list(reader)

    # Splits the length of cities list to the first 10%

    subindex = len(cities)//10

    # First data structure small_cities list

    small_cities = cities[1:subindex]

print(len(small_cities))
# Calculating the euclidean distance between the cities

# Algorithm 1

def calculate_distance(x1,y1,x2,y2):  

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

    return distance

# initialise start time to calculate total time taken 

start_time = time.time()

total_cities = len(small_cities)

current_city = small_cities[0]

small_cities.pop(0)



#set values to 0

shortest_distance = 0

closest_city = 0

total_distance = 0



#Deque used as second data structure

#Deque is used as its time to append is O(1) instead of O(n) like a list

d = deque() 

d.append('0')



#algorithm 2

#iterates throughout the list of small_cities to find the shortest distance after which it selects the closest city 

while (len(d) < total_cities):

    for i in range(0, len(small_cities)):

        current_distance = calculate_distance(float(current_city[1]),float(current_city[2]),float(small_cities[i][1]),float(small_cities[i][2]))

        if (shortest_distance == 0 or current_distance < shortest_distance):

            shortest_distance = current_distance

            closest_city = i

    # checks if city is a multiple of 10 and if it is a prime        

    if len(d) % 10 == 0 and sympy.isprime(current_city[0]) == False:

        shortest_distance = shortest_distance * 1.1

    #sets our current_city as our closest_city 

    current_city = small_cities[closest_city]

    #calculates total distance

    total_distance += shortest_distance

    #add our current_city to our deque

    d.append(current_city[0])

    #remove the closest_city from the list so that there are no repeat cities

    small_cities.pop(closest_city)

    #reset shortest_distance to 0

    shortest_distance = 0

#calculates the distance from our final city back to city 0    

start_distance = calculate_distance(float(cities[1][1]),float(cities[1][2]),float(current_city[1]),float(current_city[2]))

total_distance = total_distance + start_distance

d.append('0')

print('Final order:')

print(d)

print('Total distance:', total_distance)

print("--- %s seconds ---" % (time.time() - start_time))
import pandas as pd

cities = pd.read_csv("../input/cities.csv")