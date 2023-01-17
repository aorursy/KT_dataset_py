# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
cities = pd.read_csv('../input/cities.csv')
cities.head()
def prime_number(max_number):
    primes = [1]
    for p in range(2, max_number + 1):
        is_prime = True
        for num in range(2, int(p ** 0.5) + 1):
            if p % num == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    return primes
prime_cities = cities.loc[prime_number(len(cities)), :]
fig = plt.figure(figsize=(20,20))
plt.plot(cities.X, cities.Y, '.',color='lightblue', alpha=0.3, label='cities')
plt.plot(cities.X[0], cities.Y[0], 'o',color='fuchsia', label='North Pole')
plt.plot(prime_cities.X, prime_cities.Y, '.',color='aqua',alpha=0.5, label='prime cities')
plt.axis('off')
plt.legend();
fig.savefig('cities.png')
sub = pd.read_csv("../input/sample_submission.csv")
cities['dist'] = np.square(cities.X-cities.X[0])+np.square(cities.Y-cities.Y[0])
cities = cities.sort_values('dist')
cities.head()
path = cities['CityId'].values
num_cities = len(cities)-1
num_cities
forward = [(i+1)*2 for i in range(num_cities//2)]
backward = [(i+0)*2+1 for i in range(num_cities//2)]
fpath = path[forward]
bpath = np.array(list(reversed(path[backward])))
fpath
bpath
path1 = [0] + fpath.tolist()+bpath.tolist()+[0]
sub = pd.read_csv("../input/sample_submission.csv")
sub['Path'] = path1
sub.to_csv('simple_submission.csv', index=False)
path = path.tolist()+[0]
sub['Path'] = path
sub.to_csv('simple_submission2.csv', index=False)
