# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from math import sqrt

import matplotlib.pyplot as plt

import warnings

from collections import Counter

plt.style.use("fivethirtyeight")
#Sample Euclidean distance calculation 



a = np.array([1,2])

b = np.array([2,4])   



print("euclidean distance between a and b is: " + str(np.linalg.norm(a-b)))   #using l2 norm   
#creating a dataset



# Two classes are k, r  

# Dictionary with class name as key and point and respective point coordinates as list of lists

dataset = {"k":[[1,2],[2,3],[3,1]], "r":[[6,5],[7,7],[8,6]] }      #dictionary with class name as key and point and respective point coordinates as list of lists



new_point = [5,7]   #need to classify this point





#looping to plot the points



for i in dataset:

    for j,k in dataset[i]:

        plt.scatter(j,k,s=100,color=i)

        

#k is the black points class

#r is the red boints class



plt.scatter(new_point[0],new_point[1],s=100)



#Need to classify the blue point



def KNearestNeighbors(data,predict,k=3):                  #data input is of dictionary type

    if len(data) >=k :

        warnings.warn("K is lower than total voting classes")

    

    dist = []

    for key in data:                                      # looping to through the dictionary 

        for features in data[key]:

            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))                   #calculating the distance of nw points to all old points

            dist.append([euclidean_distance,key])

            

    votes = [item[1] for item in sorted(dist)[:k]]

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result



result = KNearestNeighbors(dataset,new_point,k=3)



print("The new point belongs to the group : " +str(result))



    