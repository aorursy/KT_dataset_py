%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns
cities = pd.read_csv('../input/cities.csv')
cities.head()
cities.shape
def is_prime(num):
    if num > 1:
        for i in np.arange(2, np.sqrt(num+1)) :
            if num % i == 0:
                return False
        
        return True
    
    return False
cities['prime'] = cities['CityId'].apply(is_prime)
sum(cities.prime)
fig, ax = plt.subplots(figsize = (10,10))
ax.scatter(x = cities.X, y = cities.Y, alpha = 0.5, s = 1)
ax.set_title('Cities chart. North Pole, prime and non prime cities', fontsize = 16)
ax.scatter(x = cities.X[0], y = cities.Y[0], c = 'r', s =12)
ax.scatter(x = cities[cities.prime].X, y = cities[cities.prime].Y, s = 1, c = 'purple', alpha = 0.3)
ax.annotate('North Pole', (cities.X[0], cities.Y[0]),fontsize =12)
ax.set_axis_off()
fig, (ax0,ax1) = plt.subplots(ncols = 2, figsize = (15,7))

ax0.set_title('Cities chart. North Pole, non prime cities', fontsize = 16)
ax0.scatter(x = cities.X[0], y = cities.Y[0], c = 'r', s =12)
ax0.scatter(x = cities[cities.prime == False ].X, y = cities[cities.prime == False].Y, s = 1, c = 'green', alpha = 0.3)
ax0.annotate('North Pole', (cities.X[0], cities.Y[0]),fontsize =12)
ax0.set_axis_off()

ax1.scatter(x = cities[cities.prime].X, y = cities[cities.prime].Y, s = 1, c = 'purple', alpha = 0.3)
ax1.set_title('Cities chart. North Pole, prime cities', fontsize = 16)
ax1.annotate('North Pole', (cities.X[0], cities.Y[0]),fontsize =12)
ax1.set_axis_off()
all_dist = distance.cdist(cities[['X','Y']][:1],cities[['X','Y']],metric = 'euclidean')
cities['dist_from_np'] = all_dist.T
cities.head()
benchmark = cities.sort_values(by = 'dist_from_np').reset_index(drop=True)
b1 = benchmark.copy()
benchmark.rename(columns = {'CityId':'Path'}, inplace = True)
benchmark = benchmark.filter(items = ['Path'],axis=1)
benchmark = benchmark.append(benchmark.iloc[0]).reset_index(drop = True)
benchmark.head() # to make it uploadable format
sns.set(rc={'figure.figsize':(10,10)})
sns.lineplot(x = b1.X, y = b1.Y, alpha = 0.3, color = 'grey')