import csv

import operator

from operator import itemgetter

from collections import deque

import math

import time

import sympy
# opened the csv file and assigned to reader, which i then assigned to a list called allCities.



with open("../input/cities/cities.csv", "r") as file:

    # eachLine = cities.readline()

    reader = csv.reader(file, delimiter=",")

#     data structure 1

    allCities = list(reader)

#     created a subgroup of allCities, created for testing purposes...

#     subindex = len(allCities) // 100

#     allCities = allCities[:subindex]



    print('candidates length', len(allCities))

    # print(allCities)

    # allCities.sort(key=itemgetter(0))

    startCity = allCities[0]

    baseCity = allCities[0]

    allCities.pop(0)

    print('candidates length', len(allCities))
# created another backupCities, so i can get the distance between the last location, and the starting location again. because the original allCities[0] gets popped, so it will be empty.

with open("../input/cities/cities.csv", "r") as file:

    # eachLine = cities.readline()

    reader = csv.reader(file, delimiter=",")

    backupCities = list(reader)

    print(len(backupCities))
totalDistance = 0

# data structure 2

final = deque()

final.append('0')

# the while loop runs while the number of lines in the list is more than 0. it calculates the distance repeatedly between the cities, until the length of the file becomes 0.

# algorithm 1

while len(allCities) > 0:

    nearestCity = -1

    nearestCityDistance = 999999

#     algorithm 2

    for i in range(len(allCities)):

        xDist = (float(allCities[i][1]) - float(baseCity[1])) ** 2

        yDist = (float(allCities[i][2]) - float(baseCity[2])) ** 2

        squareRoot = math.sqrt(xDist + yDist)

        if squareRoot < nearestCityDistance:

            nearestCityDistance = squareRoot

            nearestCity = i

    if len(final) % 10 == 0 and sympy.isprime(baseCity[0]) == False:

        nearestCityDistance = nearestCityDistance * 1.1

    # print(len(allCities))

    # time.sleep(1)

    final.append(int(allCities[nearestCity][0]))

    baseCity = allCities[nearestCity]

    allCities.pop(nearestCity)

    totalDistance += nearestCityDistance

    nearestCityDistance = 999999

#     print("City ID:", baseCity[0], "X:", baseCity[1], "Y:", baseCity[2])

#     print("No. of citites left", len(allCities))
# appending the startCity back into the list, as he needs to return back to city 0.

final.append("0")

xDist = (float(backupCities[0][1]) - float(baseCity[1])) ** 2

yDist = (float(backupCities[0][1]) - float(baseCity[2])) ** 2

squareRoot = math.sqrt(xDist + yDist)

totalDistance += squareRoot

print("The final list is: ", final)

print("")