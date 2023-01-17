# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Data Handling:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random 
import math

# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
DF_cities = pd.read_csv('../input/cities.csv')
print(DF_cities.shape)
print(DF_cities.head())
# Function to determine prime numbers:
def is_prime(n):
    if n > 2:
        i = 2
        while i ** 2 <= n:
            if n % i:
                i += 1
            else:
                return False
    elif n != 2:
        return False
    return True

#Create column in DF_cities to flag prime cities
DF_cities['IsPrime'] = DF_cities.CityId.apply(is_prime)

# Ok, let's preview the edited DF:
print(DF_cities.head(5))
DF_cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
# Add north pole in red, and much larger so it stands out
north_pole = DF_cities[DF_cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=20)
# Add in prime cities in green and slightly larger
DF_primes = DF_cities.loc[DF_cities['IsPrime'] == True]
plt.scatter(DF_primes.X, DF_primes.Y, c='green', s=0.5);
plt.show()
print(DF_cities['IsPrime'].value_counts())
sns.countplot(x='IsPrime', data = DF_cities);
# First Let's define a function to calculate distance for our solutions:
# It'd also be nice if it told us how much of a distance penalty we accrued...
def total_distance(route):
    coords = DF_cities[['X', 'Y']].values
    summed_dist = 0
    summed_extra = 0
    for i in range(1, len(route)):
        extra = 0
        begin = route[i-1]
        end = route[i]
        distance = ((coords[end,0]-coords[begin,0])**2 + (coords[end,1]-coords[begin,1])**2)**0.5
        if i%10 == 0:
            # Edit this part
            if is_prime(begin):
                pass
            else:
                distance *= 1.1
                extra = distance * 0.1
            # if begin not in PRIMES:
                # distance *= 1.1
        summed_dist += distance
        summed_extra += extra
    print('Total Distance:  ' + str(summed_dist) + '\nPenalty: ' + str(summed_extra) + '(' + str((summed_extra/summed_dist)*100) + '%)')
    return summed_dist, summed_extra
print('done!')
%%time
route = list(DF_cities['CityId'])
route.append(0)
total_distance(route)
%%time
DF_cities = pd.read_csv('../input/cities.csv') # to prevent confusion with notebooks... 
temp = DF_cities.drop(DF_cities.index[0]).sort_values(['X'], ascending = 1)

route = [0] # start at North Pole
route.extend(list(temp['CityId'])) # All other stops
route.append(0)  # End at North Pole

total_distance(route)
%%time
# Simple Nearest Neighbor Model:

# Re-load CD_cities (to avoid issues with it being modified in previous cells)
DF_cities = pd.read_csv('../input/cities.csv')

IDs = DF_cities.CityId.values
coords = np.array([DF_cities.X.values, DF_cities.Y.values]).T

# Set intial position (north pole)
position = coords[0]
route = [0]

# Remove initial position from list now that it has already been added to the route
IDs = np.delete(IDs, 0)
coords = np.delete(coords, 0, axis=0)

count = 0
# Loop through remaining cities to fill in route with the nearest remaining cities:
while len(IDs) > 0:
    # create matrix of distances from remaining cities to current city
    distance_matrix = np.linalg.norm(coords - position, axis=1)
    
    # Find Nearest City (minimum distance)
    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value
    
    # Set position for next loop and remove this city from list
    route.append(IDs[idx_min])
    position = coords[idx_min]
    IDs= np.delete(IDs, idx_min, axis=0)
    coords = np.delete(coords, idx_min, axis=0)
    
    # print out updates on progress of loop every 10000 iterations:
    if count % 10000 == 0:
        print(str(len(IDs))+ ' cities left')
    count += 1
    
# Finally, end back at the north pole:
route.append(0)

# Use the function from above to calculate the total distance travelled:
total_distance(route)
%%time
# Re-load CD_cities (to avoid issues with it being modified in previous cells)
DF_cities = pd.read_csv('../input/cities.csv')
DF_cities['IsPrime'] = DF_cities.CityId.apply(is_prime)

IDs = DF_cities.CityId.values
coords = np.array([DF_cities.X.values, DF_cities.Y.values]).T
primes = np.array(DF_cities.IsPrime)

# Set intial position (north pole)
position = coords[0]
route = [0]

# Remove initial position from list now that it has already been added to the route
IDs = np.delete(IDs, 0)
coords = np.delete(coords, 0, axis=0)
primes = np.delete(primes,0)
    
count = 0
# Loop through remaining cities to fill in route with the nearest remaining cities:
while len(IDs) > 0:
    
    # add to count:
    count += 1
    
    # create matrix of distances from remaining cities to current city
    distance_matrix = np.linalg.norm(coords - position, axis=1)
    
    if count % 10 == 0:
        idx_toPenalize = np.where(primes == False )[0]
        distance_matrix[idx_toPenalize] *= 1.10

    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value
    
    # Find Nearest City (minimum distance)
    idx_min = distance_matrix.argmin()  # np.argmin returns the index of the min value
    
    # Set position for next loop and remove this city from list
    route.append(IDs[idx_min])
    position = coords[idx_min]
    IDs= np.delete(IDs, idx_min, axis=0)
    coords = np.delete(coords, idx_min, axis=0)
    primes = np.delete(primes, idx_min, axis=0)
    
    # print out updates on progress of loop every 10000 iterations:
    if count % 10000 == 0:
        print(str(len(IDs))+ ' cities left')
    
# Finally, end back at the north pole:
route.append(0)

# Use the function from above to calculate the total distance travelled:
total_distance(route)
output = pd.DataFrame({'Path': route})
output.to_csv('submission.csv', index=False)

print('Output data file saved')






