# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import math
import doctest
from itertools import permutations
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Importing data
cities_df = pd.read_csv("../input/cities.csv") 
submit = pd.read_csv("../input/sample_submission.csv")
print("City data loaded...") # Confirm

print(cities_df.shape)
print(cities_df.head())

cityId = cities_df["CityId"]
x = cities_df["X"]
y = cities_df["Y"]
North_Pole = x[0],y[0]
print("The North Pole is located at: " + str(North_Pole) + ".")
print("This will be our starting location.")

all_other_cities = x[1:],y[1:]
print("Sample of some of the other cities: \n" + str(x[1:5]) + str(y[1:5]) + ".")

#Finding primes
def is_prime(n):
    if n == 1:
        return 0 #False
    if n == 2:
        return 1 #True
    if n > 2 and n % 2 == 0:
        return 0 #False
    
    count_prime = 0
    div = math.floor(math.sqrt(n))
    for d in range(3, 1 + div, 2):
        if n % d == 0:
            return 0 #False
    return 1 #True

cities_df['is_prime'] = cities_df.CityId.apply(is_prime)
count_prime_cities = np.sum(cities_df['is_prime'])

# Detailing cities
print("There are " + str(len(cities_df) - 1) + " cities.") 
print("There are " + str(count_prime_cities) + " prime cities.")
print("There are " + str(len(cities_df) - 1 - count_prime_cities) + " that are not prime and santa has to reach.")

# Plot
fig = plt.figure(figsize=(15,15))
plt.scatter(x, y, marker='o', s=1, color='b', linewidths=0)
plt.scatter(x[0],y[0], marker='o', color='r', linewidths=1)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Reindeer Scatter")
plt.show()

cities_df["adding_distance"] = x + y
print(cities_df.head())
submit['Path'] = np.append(0, cities_df.sort_values('adding_distance')['CityId'].values)
submit.loc[24740] = 197338; submit.loc[197769] = 0 #First and last cities are North Pole
print(submit['Path'])
submit.to_csv('submission.csv', index=False)
