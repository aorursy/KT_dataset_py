import csv # import csv"

from collections import deque # to use data structure deque

import math   # to use math.sqrt 

import sympy  # to check if a number is a prime number

import time    # to calculate time taken for the program
# read cities.csv

with open('../input/cities/cities.csv', 'r') as file:

    file_read = csv.reader(file)

    # put cities and coordinates into a list

    cities = list(file_read)

    # calculate 10% of the list

    subnum = len(cities)//10

    # only use the first 10% of the list

    sub_cities = cities[1:subnum]

print(len(sub_cities))
"""Calculate the distance between cities"""



def calculate_distance(x1, y1, x2, y2):  

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

    return distance

# set the initial time

start_time = time.time()

all_cities = len(sub_cities)

current_city = sub_cities[0]

sub_cities.pop(0)



shortest_distance = 0

closest_city = 0



# deque for insertion and deletion at both ends 

d = deque()

d.append('0')

total_distance = 0



while (len(d) < all_cities):

    # loops through the list of sub_cities to find the shortest distance to travel and select that city 

    for i in range(0, len(sub_cities)):

        current_distance = calculate_distance(float(current_city[1]),float(current_city[2]),float(sub_cities[i][1]),float(sub_cities[i][2]))

        if (shortest_distance == 0 or current_distance < shortest_distance):

            shortest_distance = current_distance

            closest_city = i

    # check if cities are multiples of 10 and if their number is a prime

    if len(d) % 10 == 0 and sympy.isprime(current_city[0]) == False:

        shortest_distance = shortest_distance * 1.1

            

    current_city = sub_cities[closest_city]  # sets current_city as closest_city

    d.append(current_city[0])  # puts the city into deque

    total_distance += shortest_distance   # add shortest distance into total

    sub_cities.pop(closest_city)  # pop the closest city from list to avoid repetition

    shortest_distance = 0  

start_distance = calculate_distance(float(cities[1][1]),float(cities[1][2]),float(current_city[1]),float(current_city[2]))

total_distance = total_distance + start_distance

d.append('0')

print('Final order:')

print(d)

print('Total distance:', total_distance)

print("Time taken is", time.time() - start_time, "seconds.")
import pandas as pd

cities = pd.read_csv("../input/cities.csv")