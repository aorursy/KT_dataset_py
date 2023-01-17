# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import glob

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

color = sns.color_palette()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/auto-mpg.csv')

data.head()
# Check the type of data values

data.info()
# How many different types of cars are there?

unique_cars = set(data['car name'].values)

print(len(unique_cars))
# Check for duplicates

data[data.duplicated()]
# There are NaN values in the horespower column which are represented by `?` due to which it is of object type

# Replace the NaN values with the min horsepower value 

horse_power_values = [int(val) for val in data['horsepower'].values if val!='?']

min_horsepower = min(horse_power_values)

print("Minimum horsepower value : ", min_horsepower)



data['horsepower'] = data['horsepower'].apply(lambda x : int(x) if x!='?' else min_horsepower)
# Next, let's check the distribution of different columns

f, ax = plt.subplots(3,2, figsize=(20,20))

columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']



for i, col in enumerate(columns):

    ax[i//2, i%2].scatter(range(len(data)), np.sort(data[col].values), alpha=0.8)

    ax[i//2, i%2].set_xlabel('Index')

    ax[i//2, i%2].set_ylabel('Values')

    ax[i//2, i%2].set_title(col)
# Let's check the cars for which there are multiplr entries

from collections import defaultdict

import operator



car_map = defaultdict(int)



for car in data['car name'].values:

    car_map[car] += 1

    

car_map = dict(sorted(car_map.items(), key=operator.itemgetter(1), reverse=True))

car_map = [(k, car_map[k]) for k in car_map.keys() if car_map[k] > 1]

print(car_map)

print( " ")

print("Total number of multiple entries : ", len(car_map))