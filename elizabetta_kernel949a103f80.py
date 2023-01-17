%matplotlib notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.utils.extmath

from math import sqrt
train_table = pd.read_csv('train.csv')
def take_nums(s):

    nums = [ int(i) for i in s.split(' ')[1:] ]

    return nums

    
train_table['visits'] = train_table['visits'].apply(take_nums)
def factorial(n):

    res = 1

    for i in range(1, n+1): 

        res *= i

    return res



weights = [sqrt(i) for i in range(1, 1100)]
week_days = []

for i in train_table['visits']:

    week_days.append([((j-1)%7+1) for j in i])

weight_days = []

for i in train_table['visits']:

    weight_days.append([weights[j-1] for j in i])
res = []

for i in range(0, len(train_table['visits'])):

    temp = sklearn.utils.extmath.weighted_mode(week_days[i], weight_days[i])

    res.append(temp) 
res = [' ' + str(int(x[0][0])) for x in res]
result = pd.DataFrame(columns=['id', 'nextvisit'])

result['id'] = train_table['id']

result = result.assign(nextvisit=res)

result.to_csv('solution.csv', index=False, sep=',')
res
