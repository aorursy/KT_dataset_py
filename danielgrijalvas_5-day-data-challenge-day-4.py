# 5 Day Data Challenge, Day 4

# Visualize categorical data



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from collections import Counter



# read movies.csv 

file = '../input/movies.csv'

data = pd.read_csv(file, encoding='latin1')



data.head()
# dictionary containing the frequency for each genre

counter = Counter(data['genre'])

genres = list(counter.keys())

freq = list(counter.values())



# positions in the Y-axis for each genre

y_pos = np.arange(len(genres))



# show plot, assign a title and all that stuff

plt.barh(y_pos, freq)

plt.yticks(y_pos, genres)

plt.xlabel('Frequency')

plt.title('Genres of the most popular films of the last 30 years')
# BONUS: let's visualize movie revenue over the years



# list of years in the dataset (1986-2016)

years = sorted(pd.unique(data['year']))



# calculate average gross for a movie in each year

avg_gross = [np.mean(data[data['year']==year]['gross'])/1000000 for year in years]



# plot and stuff

plt.plot(years, avg_gross)

plt.ylabel('Revenue (in millions)')

plt.xlabel('Year')

plt.title('Average movie revenue over the years')