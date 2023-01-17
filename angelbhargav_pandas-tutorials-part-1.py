# Importing Libraries

import pandas as pd

import numpy as np
s = pd.Series()

print(s)
# Without Index

data = np.array(['a','b','c','d'])

s = pd.Series(data)

print(s)
# with Index

data = np.array(['a','b','c','d','e'])

index = [100,101,102,103,104]

s = pd.Series(data,index = index)

s
# Observe − Index order is persisted and the missing element is filled with NaN.

data = {'a':0,'b':2.5,'d':9}

s = pd.Series(data,index = ['b','a','r','d','t'])

s
s = pd.Series(5,index = [1,2,3,4,5,6])

s