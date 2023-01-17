import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import operator



# Pull out the fields we care about

data = pd.read_csv("../input/zomato.csv")[['rate', 'votes', 'cuisines']]



# Find all the types of cuisine

cuisines = {}

for i, j in data.iterrows():

    

    # Each restaurant can have multiple types of cuisine

    types = [x.strip() for x in str(j[2]).split(',')]

    for cuisine in types:

        cuisines[cuisine] = cuisines[cuisine] + 1 if cuisine in cuisines.keys() else 1



# Bar chart!

x_pos = np.arange(len(cuisines))

plt.bar(x_pos, list(cuisines.values()))

plt.show()
sorted_cuisines = sorted(cuisines.items(), key=operator.itemgetter(1), reverse=True)

sorted_cuisines = sorted_cuisines[:40]



# Bar chart!

plt.subplots(figsize=(20, 6.4))

x_pos = np.arange(len(sorted_cuisines))

plt.bar(x_pos, [x[1] for x in sorted_cuisines], width=0.8)

plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)

plt.show()

# New counters for the same types

cuisine_ratings = {}

for cuisine in sorted_cuisines:

    cuisine_ratings[cuisine[0]] = 0

    

# Count em up

for i, j in data.iterrows():

    

    # Parse the rating - invalid ratings get mean score which could affect emerging cuisines

    try:

        rating = float(str(j[0])[:str(j[0]).find('/')])

    except ValueError:

        rating = 2.5

    

    # Each restaurant can have multiple types of cuisine

    types = [x.strip() for x in str(j[2]).split(',')]

    for cuisine in types:

        if cuisine in cuisine_ratings.keys():

            cuisine_ratings[cuisine] += rating

            

# Average em out

for cuisine in sorted_cuisines:

    cuisine_ratings[cuisine[0]] /= cuisine[1]

    

# Bar chart!

plt.subplots(figsize=(20, 6.4))

x_pos = np.arange(len(sorted_cuisines))

plt.bar(x_pos, [cuisine_ratings[y] - 3 for y in [x[0] for x in sorted_cuisines]], width=0.8, bottom=3)

plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)

plt.show()

# New counters for the same types

cuisine_votes = {}

for cuisine in sorted_cuisines:

    cuisine_votes[cuisine[0]] = 0

    

# Count em up

for i, j in data.iterrows():

    

    # Each restaurant can have multiple types of cuisine

    types = [x.strip() for x in str(j[2]).split(',')]

    for cuisine in types:

        if cuisine in cuisine_votes.keys():

            cuisine_votes[cuisine] += j[1]



# Bar chart!

plt.subplots(figsize=(20, 6.4))

x_pos = np.arange(len(sorted_cuisines))

plt.bar(x_pos, [cuisine_votes[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)

plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)

plt.show()

# Bar chart!

plt.subplots(figsize=(20, 6.4))

x_pos = np.arange(len(sorted_cuisines))

plt.bar(x_pos, [(cuisine_ratings[y] - 3) * cuisine_votes[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)

plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)

plt.show()
# Bar chart!

plt.subplots(figsize=(20, 6.4))

x_pos = np.arange(len(sorted_cuisines))

plt.bar(x_pos, [((cuisine_ratings[y] - 3) * cuisine_votes[y])/cuisines[y] for y in [x[0] for x in sorted_cuisines]], width=0.8)

plt.xticks(x_pos, [x[0] for x in sorted_cuisines], rotation=45)

plt.show()