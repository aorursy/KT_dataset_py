import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

data.head(15)
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, fmt= '.2f',ax=ax,cmap="YlGnBu")

plt.show()
print(data.shape) #Number of columns and rows of data
print(data.columns) #columns data types

print(data.dtypes)
print(data.isnull().sum().sort_values(ascending=False))
data.describe()