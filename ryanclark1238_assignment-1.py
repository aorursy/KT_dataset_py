# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.spatial import distance 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
cities=pd.read_csv('../input/cities.csv',low_memory=False)

cities.head()
#Make a subset of 10% of the total dataset

sub1=cities[cities.CityId<19776]

sub2=sub1.copy()

sub2.head()
#calculate prime numbers and append column to data

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



sub2['Prime']=sub2.CityId.apply(is_prime)

print(sub2['Prime'].value_counts())
#Basic algorithm that arrages X values and returns a path of CityId's

sub2=sub2.sort_values(by=['X'])

path=[]

for i in sub2['CityId']:

    path.append(i)

#prints first 10 steps of path

for i in range(10):

    print(path[i])
#caluclate total euclidean distance of a given path 

def distancecal(path):

    total=0

    num=0

    previous=path.pop()

    for i in path:

        c=[(sub2.X[i],sub2.Y[i])]

        p=[(sub2.X[previous],sub2.Y[previous])]

        d=distance.cdist(c,p,'euclidean')

        dist=d[0,0]

        if num%10==0:

            if sub2.Prime[i]==False:

                dist*=1.1    

        total+=dist

        previous=i

        num+=1

    return total

print(distancecal(path))
#formats path into submission format

submission=pd.DataFrame(path)

submission.to_csv('submission.csv',index=False)
