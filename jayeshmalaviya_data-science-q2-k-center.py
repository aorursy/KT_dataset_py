# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import preprocessing

import random

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df =pd.read_csv("/kaggle/input/ccgeneral/CC-GENERAL.csv")#.drop(columns='CUST_ID')

#missing = df.isna().sum()

#print(missing)

df1 = df.dropna()



l = df1['CUST_ID'].values

df1 = df1.drop(columns='CUST_ID')

##### --------------------------------------------------------- Normalization of data into [0,1]

x = df1.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df1 = pd.DataFrame(x_scaled)

df1.insert(0, "CUST_ID", l , True)



#missing1 = df1.isna().sum()

#print(missing1)
def random_sample_data(data, size_sample):

    

    if(type(size_sample) == float):

        size_sample = round(size_sample * len(data))

    np.random.seed(42)

    size_sample_index = np.random.choice(data.index.tolist(), size=size_sample)

    sample_data = data.loc[size_sample_index]

    return sample_data
def distance(nodes, centers):

   

    minimum_node = None

    maximum = 0

    maximum_node = None



    for n in nodes:

      minimum = np.inf

      for c in centers:

        dist = np.linalg.norm(pd.Series(n[1:]).values - pd.Series(c[1:]).values)

        print("current minimum {} vs {}".format(minimum, dist))

        if dist < minimum:

          minimum = dist

          minimum_node = n

      if minimum > maximum:

        maximum = minimum 

        maximum_node = minimum_node

    #cust_id = maximum_node[0]

    print(maximum, maximum_node)

    return maximum_node, maximum
def objective_function(points, centers):

    max_node, max_dist = distance(points, centers)

    print("\nMaximum Radious Cluster is : {} and Radious is: {}".format(max_node[0], max_dist))
def K_center(data, no_center):

      data_small = random_sample_data(data, 0.9)

      k = no_center

      new_header = data_small.iloc[0] 

      data_s = data_small[1:] 

      data_s.columns = new_header



      Row_list =[] 



      for i in range((data_s.shape[0])): 

          Row_list.append(list(data_s.iloc[i, :])) 

      #print(Row_list)

      s = []

      s.append(Row_list[0])

      #print("Before len of row list: {}  ".format(len(Row_list)))

      del Row_list[0]

      #np.random.seed(42)

      #first = data_s.iloc[int(np.random.randint(len(data_s), size=1))]

      #s.append(first.values.tolist())

      print(s)

      count = 1

      while(count <= k):

          #print("At count : {} len of row list: {}  ".format(count, len(Row_list)))

          max_node, max_dist= distance(Row_list, s)

          #data_s.drop(max, axis = 0, inplace = True)

          index = Row_list.index(max_node)

          del Row_list[index]

          s.append(max_node)

          count += 1

          #print("End of count : {} len of row list: {}  ".format(count, len(Row_list)))

      #print(s)

      objective_function(Row_list,s)

      #print("End of count : {} len of row list: {}  ".format(count, len(Row_list)))
K_center(df1, 2)