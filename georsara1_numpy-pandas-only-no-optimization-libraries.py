#Import libraries
import pandas as pd
import numpy as np
#Read data
df = pd.read_csv('../input/cities.csv')
submission = pd.read_csv('../input/sample_submission.csv')

df.head()
#Initialize variables
number_of_cities = df.shape[0]
non_traversed_cities = np.array((range(df.shape[0])))
current_city = 0

iteration = 1
traversed = []
total_distance = []

print('Number of cities to traverse:', number_of_cities)
#Find primes
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
prime_cities = df.CityId.apply(is_prime)
non_prime_index = prime_cities[prime_cities==False].index.values

print('Number of Prime cities:', sum(prime_cities))
#Find next closer city and add to traversed. Repeat until all cities traversed
while len(traversed)<number_of_cities:

    #print some info for debugging purposes
    # print('Iteration number:,', iteration)
    # print('Unique cities visited:', len(np.unique(traversed)) + 1)
    # print('Current city:', current_city)
    # print(' ')

    # create distance matrix of current city to remaining cities
    df_remain = df.loc[non_traversed_cities,:].reset_index(drop=True)
    coords = np.array([df_remain.X.values, df_remain.Y.values]).T
    current_city_idx = df_remain.index[df_remain['CityId']==current_city].values[0]
    df_remain['distances'] = np.linalg.norm(coords-coords[current_city_idx], axis = 1)

    # Make the distance of current city from self bigger than any other value to avoid selection
    df_remain['distances'][df_remain['CityId']== current_city] = df_remain['distances'].max() + 1

    #Add penalty if the city in every ten iterations is not prime
    if iteration%10 == 0:
        #Find the positions of remaining non prime cities
        non_prime_idx = df_remain['CityId'].isin(non_prime_index)
        df_remain['distances'].loc[non_prime_idx] = df_remain['distances'] * 1.1
        print('Number of cities tralleved:', len(traversed),
              'Percentage complete:', np.round(len(traversed)/number_of_cities*100,4), '%')


    #Find next closer city
    next_city = df_remain['CityId'][df_remain['distances']==df_remain['distances'].min()].values[0]

    # #Sanity check
    # if next_city in traversed:
    #     print('This city has been visited')
    #     break

    #Add current city to traversed and remove from non_traversed
    traversed.append(current_city)
    non_traversed_cities = np.delete(non_traversed_cities, np.where(non_traversed_cities==current_city))

    #Add distance travelled to total_distance
    total_distance.append(df_remain['distances'].min())

    #Set next city as the current city to repeat the process
    current_city = next_city
    iteration +=1
#Add the return to north pole
traversed.append(0)

#Write to file to submit
submission['Path'] = traversed
submission.to_csv('santa_path.csv', index = False)
submission.head()