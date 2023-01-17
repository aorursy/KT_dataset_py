import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import os
print(os.listdir("../input"))

pd.options.mode.chained_assignment = None  # default='warn'

# calculate total distance of the path
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance + \
            np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) * \
            (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance
santa_cities = pd.read_csv('../input/cities.csv')
santa_cities.head()
santa_cities.describe()
def sieve_of_eratosthenes(n):
    n = int(n)
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)

# find cities that are prime numbers
prime_cities = sieve_of_eratosthenes(max(santa_cities.CityId))
santa_cities['prime'] = prime_cities

# add a few columns
santa_cities = pd.concat([santa_cities,pd.DataFrame(columns=['next_city',
                                                             'next_city_distance'
                                                            ]
                                                   )],sort=False)
santa_cities[['next_city','next_city_distance']] = santa_cities[['next_city','next_city_distance']].astype(float)

santa_cities.head(10)
# find the optimal nearest neighbor for each row
# exclude index row 0 to ensure that doesn't get assigned until the end
santa_cities_to_analyze = santa_cities.iloc[1:, 1:3]
nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(santa_cities_to_analyze)
nn_distances, nn_indices = nbrs.kneighbors(santa_cities_to_analyze)

# loop through each layer of nearest neighbors
for i in range(1,nn_indices.shape[1]):
    # only get rows where next_city is null
    santa_cities_remaining = santa_cities.loc[santa_cities['next_city'].isnull()]
    santa_cities_remaining = pd.concat([santa_cities_remaining,
                                        pd.DataFrame(columns=['NN_city','NN_distance'])],
                                       sort=False)
    # get potential nearest neighbors in this layer
    santa_cities_remaining['NN_city'] = pd.Series(data=nn_indices[:,i])
    santa_cities_remaining['NN_distance'] = pd.Series(data=nn_distances[:,i])
    
    # set next_city to nearest neighbor in this layer
    santa_cities['next_city'].update(santa_cities_remaining['NN_city'])
    santa_cities['next_city_distance'].update(santa_cities_remaining['NN_distance'])
    
    # sort by the distance of nearest neighbors to favor shorter distances
    santa_cities = santa_cities.sort_values(by=['next_city_distance'],ascending=True)
    
    # mark all duplicates of next_city except for the first with the shortest distance
    santa_cities['duplicate'] = santa_cities.duplicated(subset='next_city', keep="first")
    # clear data for all duplicates
    santa_cities.at[santa_cities['duplicate'] == True,'next_city'] = np.nan
    santa_cities.at[santa_cities['duplicate'] == True,'next_city_distance'] = np.nan
    
    # TO DO - determine if I can remove this section of code with the merge
    # check for one step recursive links between cities
    santa_cities_test = santa_cities[['CityId','next_city']]
    santa_cities_test.rename(columns={'CityId': 'next_city', 'next_city': 'compare_city'}, inplace=True)
    santa_cities = santa_cities.reset_index().merge(santa_cities_test,
                                                    on='next_city',
                                                    how='left').set_index('index')
    santa_cities.at[santa_cities['compare_city'] == santa_cities['CityId'],'next_city'] = np.nan
    santa_cities.at[santa_cities['compare_city'] == santa_cities['CityId'],'next_city_distance'] = np.nan
    santa_cities.drop(columns=['compare_city'], inplace=True)
    
    # check for deeper recursive links
    santa_cities = pd.concat([santa_cities,pd.DataFrame(columns=['city_order',
                                                                 'next_city_order']
                                                       )],sort=False)
    
    santa_cities.at[0,'next_city'] = 146845 # ensure beginning of path is set
    santa_cities.at[0,'next_city_distance'] = 7.358130
    santa_recursion = santa_cities.loc[~santa_cities['next_city'].isnull()]
    city_to_travel_next = 0
    for i in range(1,len(santa_recursion)):
        current_city = city_to_travel_next
        santa_cities.at[santa_cities['CityId'] == current_city,'city_order'] = i
        santa_cities.at[santa_cities['next_city'] == current_city,'next_city_order'] = i
        try:
            city_to_travel_next= int(santa_cities.at[current_city,'next_city'])
        except:
            break
    
    santa_cities.at[santa_cities['next_city_order'] < santa_cities['city_order'],'next_city'] = np.nan
    santa_cities.at[santa_cities['next_city_order'] < santa_cities['city_order'],'next_city_distance'] = np.nan
    santa_cities.drop(columns=['city_order','next_city_order'], inplace=True)
    
    # ensure 0 doesn't get marked as a next_city and is left for the very end
    santa_cities.at[santa_cities['next_city'] == 0,'next_city'] = np.nan
    santa_cities.at[santa_cities['next_city'] == 0,'next_city_distance'] = np.nan
    
    # sort index to normal ordering again
    santa_cities.sort_index(inplace=True)

# show number of remaining cities that still need a next_city
print(santa_cities['next_city'].isnull().sum())
if 'duplicate' in santa_cities.columns:
    santa_cities.drop(columns=['duplicate'],inplace=True)
santa_cities.index.name = None
santa_cities.head(10)
print("Total optimized nn attempt distance so far is {:,}".format(santa_cities['next_city_distance'].sum()))
#santa_cities.nunique()

#santa_path = []
#city_to_travel_next = int(0)
#santa_path_cities = santa_cities.loc[~santa_cities['next_city'].isnull()]

#for i in range(1,len(santa_cities)): #len(santa_cities)
#    current_city = city_to_travel_next
#    santa_path.append(current_city)
#    city_to_travel_next = int(santa_cities.iloc[current_city,4])

#santa_path.append(0)
#len(santa_path)

# check for recursion in travel path
#print(len(set(santa_path)))

#print("Total optimized nn attempt distance is {:,}".format(total_distance(santa_cities,santa_path)))
def nearest_neighbour():
    cities = pd.read_csv("../input/cities.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    return path

path_nn = nearest_neighbour()
print("Total nn distance is {:,}".format(total_distance(santa_cities,path_nn)))
pd.DataFrame({'Path':path_nn}).to_csv('nearest_neighbor_cities.csv',index = False)