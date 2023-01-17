# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 

data = [[6, 2], [4, 2], [6, 4], [8, 2]]

data





a = np.array(data)

np.std(a, axis=0)

np.mean(a, axis=0)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(data)

print(scaler.mean_)

z=scaler.transform(data)

z