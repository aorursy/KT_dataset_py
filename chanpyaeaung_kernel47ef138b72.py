import csv # imported to handle csv files with "with"

from collections import deque # to use data structure deque

import math   # to use math.sqrt 

import sympy  #to check number is prime or not

import time    # to calculate time taken for the program
# open the cities.csv in only read

with open("../input/cities.csv", 'r') as file:

    file_read = csv.reader(file)

    # to put citiies and coordinates into list

    cities = list(file_read)

    # calculate 10% of the list

    subnum = len(cities)//10

    # to use the first 10% of the list, (first data structure, list)

    sub_cities = cities[1:subnum]

print(len(sub_cities))
# function to calculate straight distance between cities

# algorithm 2

def calculate_distance(x1,y1,x2,y2):  

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

    return distance

#to know the starting time

start_time = time.time()

all_cities = len(sub_cities)

current_city = sub_cities[0]

sub_cities.pop(0)



shortest_distance = 0

close_city = 0



# deque is used it can be used for insertion and deletion at both ends 

d = deque()

d.append('0')

total_distance = 0



while (len(d) < all_cities):

    # algorithm 1

    # loops through the list of sub_cities to find the shortest distance to travel and select that city 

    for i in range(0, len(sub_cities)):

        current_distance = calculate_distance(float(current_city[1]),float(current_city[2]),float(sub_cities[i][1]),float(sub_cities[i][2]))

        if (shortest_distance == 0 or current_distance < shortest_distance):

            shortest_distance = current_distance

            close_city = i

    # check if cities are multiples of 10 and whether number is prime

    if len(d) % 10 == 0 and sympy.isprime(current_city[0]) == False:

        shortest_distance = shortest_distance * 1.1

            

    current_city = sub_cities[close_city]   # sets our current_city as close_city

    d.append(current_city[0]) # puts that city into deque

    total_distance += shortest_distance   #shortest distance combined into total

    sub_cities.pop(close_city)     # pop the closest city from list so that we do not have the same city twice

    shortest_distance = 0  

start_distance = calculate_distance(float(cities[1][1]),float(cities[1][2]),float(current_city[1]),float(current_city[2]))

total_distance = total_distance + start_distance

d.append('0')

print('Final order:')

print(d)

print('Total distance:', total_distance)

print("Time taken is", time.time() - start_time, "seconds.")