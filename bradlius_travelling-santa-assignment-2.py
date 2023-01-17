# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math # Math functions

import matplotlib.pyplot as plt # Plotting for graphs

import seaborn as sns

from time import time

from matplotlib import collections  as mc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# retrieve dataset

df1 = pd.read_csv('../input/cities.csv')

# function to check for a prime number

def isPrime(n):

    primes = [True for i in range(n+1)] 

    primes[0] = False 

    primes[1] = False

    for i in range(2,int(np.sqrt(n)) + 1):

        if primes[i]:

            k = 2

            while i*k <= n:

                primes[i*k] = False

                k += 1

    return(primes)

nb_cities = max(df1.CityId)

primes = np.array(isPrime(nb_cities)).astype(int)

df1['P'] = isPrime(max(df1.CityId))

df2 = isPrime(max(df1.CityId))

df1.head(10)
# function to calculate the distance between two cities



def distanceMeasure(x,y):

    x1 = (df1.X[x] - df1.X[y]) ** 2

    x2 = (df1.Y[x] - df1.Y[y]) ** 2

    return np.sqrt(x1 + x2)
def total_distance(path):

    distance = [distanceMeasure(path[x], path[x+1]) + 0.1 * distanceMeasure(path[x], path[x+1])

    if (x+1)%10 == 0 and df1.P[path[x]] == False else distanceMeasure(path[x], path[x+1]) for x in range(len(path)-1)]

    return np.sum(distance)
dumbest_path = df1['CityId'].values

dumbest_path =  np.append(dumbest_path,0)

print('Total distance with the worst case path is '+ "{:,}".format(total_distance(dumbest_path)))
X=[]

for x in range(max(df1.CityId+1)):

    X.append(df1['X'][x])  

Y=[]

for x in range(max(df1.CityId+1)):

    Y.append(df1['Y'][x])

path=[]

for x in range(1,max(df1.CityId)+1):

    path.append(x)
def merge(s1, s2, S):

    i = j = k = 0

    while i < len(s1) and j < len(s2): 

        if Y[s1[i]] < Y[s2[j]]: 

            S[k] = s1[i] 

            i+=1

        else: 

            S[k] = s2[j] 

            j+=1

        k+=1

        

            

def merge_sort(S):

    if len(S) > 1:

        mid = len(S) // 2

        s1 = S[0:mid]

        s2 = S[mid:]

    

        merge_sort(s1)

        merge_sort(s2)

        

        merge(s1, s2, S)

   

    
mergesort_path=[]

for x in range(1,max(df1.CityId)+1):

        mergesort_path.append(x)

merge_sort(mergesort_path)
mergesorted_path=[0]

for i in range(len(mergesort_path)):

    mergesorted_path.append(mergesort_path[i])

mergesorted_path.append(0)
print('Total distance with Merge sorted cities using Y is: '+ "{:,}".format(total_distance(mergesorted_path)))
import sys

sys.setrecursionlimit(500000)
def quick_sort(A):

    quick_sort2(A, 0, len(A)-1)
def quick_sort2(A, low, hi):

        if low < hi:

            p = partition(A, low, hi)

            quick_sort2(A, low, p - 1)

            quick_sort2(A, p + 1, hi)
def get_pivot(A, low, hi):

    mid = (hi + low) // 2

    s = sorted([A[low], A[mid], A[hi]])

    if s[1] == A[low]:

        return low

    elif s[1] == A[mid]:

        return mid

    return hi


def partition(A, low, hi):

    pivotIndex = get_pivot(A, low, hi)

    pivotValue = A[pivotIndex]

    A[pivotIndex], A[low] = A[low], A[pivotIndex]

    border = low



    for i in range(low, hi+1):

        if X[A[i]] < X[pivotValue]:

            border += 1

            A[i], A[border] = A[border], A[i]

    A[low], A[border] = A[border], A[low]



    return (border)
quicksort_path=[]

for x in range(1,max(df1.CityId)+1):

        quicksort_path.append(x)
quick_sort(quicksort_path)
quicksorted_path=[0]

for x in range(len(quicksort_path)):

    quicksorted_path.append(quicksort_path[x])

quicksorted_path.append(0)
print('Total distance with the insertion Sorted city based on X path is '+ "{:,}".format(total_distance(quicksorted_path)))
from random import randint

tree_path = []

class node:

    def __init__(self, value = None):

        self.value = value

        self.left = None

        self.right = None

        

class binary_search_tree:

    def __init__(self):

        self.root = None

        

    def insert(self, value):

        if self.root == None:

            self.root = node(value)

        else:

            self._insert(value, self.root)

    

    

    def _insert(self, value, cur_node):

        if Y[value] <= Y[cur_node.value]:

            if cur_node.left == None:

                cur_node.left = node(value)

            else:

                self._insert(value, cur_node.left)

        elif X[value] >= X[cur_node.value]:

            if cur_node.right == None:

                cur_node.right = node(value)

            else:

                self._insert(value, cur_node.right)

        else:

            tree_path.append(path[cur_node.value])



    

    

    def print_tree(self):

        if self.root != None:

            self._print_tree(self.root)

    

    

    def _print_tree(self, cur_node):

        if cur_node != None:

            self._print_tree(cur_node.left)

            tree_path.append(path[cur_node.value])

            self._print_tree(cur_node.right)    



                

def fill_tree(tree):

    for x in range(len(path)-1):

        cur_elem = path[x]

        tree.insert(cur_elem)

    return tree



tree = binary_search_tree()

tree = fill_tree(tree)

tree.print_tree()



print('Total distance with the Binary Tree sorting city based on X path is '+ "{:,}".format(total_distance(tree_path)))

            



penalization = 0.1 * (1 - primes) + 1

def dist_matrix(coords, i, penalize=False):

    begin = np.array([df1.X[i], df1.Y[i]])[:, np.newaxis]

    mat =  coords - begin

    if penalize:

        return np.linalg.norm(mat, ord=2, axis=0) * penalization

    else:

        return np.linalg.norm(mat, ord=2, axis=0)

    

    

def get_next_city(dist, avail):

    return avail[np.argmin(dist[avail])]   

    

    

coordinates = np.array([df1.X, df1.Y])   

current_city = 0     

left_cities = np.array(df1.CityId)[1:]    

greed_path = [0]    

stepNumber = 1   



t0 = time()



while left_cities.size > 0:

    if stepNumber % 10000 == 0: #We print the progress of the algorithm

        print(f"Time elapsed : {time() - t0} - Number of cities left : {left_cities.size}")

    favorize_prime = (stepNumber % 10 == 9)

    distances = dist_matrix(coordinates, current_city, penalize=favorize_prime)

    current_city = get_next_city(distances, left_cities)     # Get the closest city and go to it

    left_cities = np.setdiff1d(left_cities, np.array([current_city]))

    greed_path.append(current_city)

    stepNumber += 1

    

    

    

    

    

    

    

    

    

    

    

    
print(f"Loop lasted {(time() - t0) // 60} minutes ")
greed_path.append(0)

print(len(greed_path))

print('Total distance with the Binary Tree sorting city based on X path is '+ "{:,}".format(total_distance(greed_path)))
def submission():

    dict = {'Path': tree_path}  

    df = pd.DataFrame(dict) 

    #write data from dataframe to csv file

    df.to_csv('Final_Submission.csv', index=False)
submission()