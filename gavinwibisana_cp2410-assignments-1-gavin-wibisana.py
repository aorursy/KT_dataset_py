#importing the functions



import csv

import math

import sympy

import time
#opening the csv files in read only mode



with open('cities.csv', 'r') as file:

    read_file = csv.reader(file)

    cities = list(read_file)

    subname = len(cities)//10

    partcities = cities[1:subname]

print(len(partcities))
#function to calculate distance from point to point in the city

def distance_calculate(x1,y1,x2,y2):  

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

    return distance
#telling the program what variable must be use

time_start = time.time()

cities_total = len(partcities)

cities_current = partcities[0]

partcities.pop(0)



distance_shortest = 0

city_close = 0



city_visited = ["0"]

distance_total = 0



#program to calculate the shortest distance

while (len(city_visited) < cities_total):

    for i in range(0, len(partcities)):

        distance_current = distance_calculate(float(cities_current[1]),float(cities_current[2]),float(partcities[i][1]),float(partcities[i][2]))

        if (distance_shortest == 0 or distance_current < distance_shortest):

            distance_shortest = distance_current

            city_close = i

    if len(city_visited) % 10 == 0 and sympy.isprime(cities_current[0]) == False:

        distance_shortest = distance_shortest * 1.1

    #print the closest city

    print('\nClosest city is', partcities[city_close][0], ':', distance_shortest, '\n')

    cities_current = partcities[city_close]

    city_visited.append(cities_current[0])

    distance_total += distance_shortest

    #output the total distance

    print("Total Distance are: ", distance_total)

    partcities.pop(city_close)

    distance_shortest = 0

#calculate the distance

distance_start = distance_calculate(float(cities[1][1]),float(cities[1][2]),float(cities_current[1]),float(cities_current[2]))

distance_total = distance_total + distance_start

city_visited.append('0')

print('Final order are:')

print(city_visited)

print('Total distance are:', distance_total)

print("Time taken is", time.time() - time_start, "seconds.")