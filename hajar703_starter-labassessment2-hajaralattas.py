from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd

from fancyimpute import KNN  



# Lab_Assessment_2.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/Lab_Assessment_2.csv', delimiter=',')

df1.dataframeName = 'Lab_Assessment_2.csv'



nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')









list(df1)
df1.dtypes

df1.info()
df1.describe()

convert_attribute = {'Salary': float}

df1 = df1.astype(convert_attribute)

print(df1.dtypes)


print(df1.columns)

print(pd.isnull(df1).sum())

dataAverage = df1.copy()

dataKnn = df1.copy()

mean_value=dataAverage['Age'].mean()

dataAverage['Age']=dataAverage['Age'].fillna(mean_value)

dataAverage.isnull().sum()



dataKNN = df1.copy()

data_cols = list(dataKNN)

dataKNN = dataKNN[['Salary', 'Age']]

dataKNN = pd.DataFrame(KNN(k=1).fit_transform(dataKNN))








df1.isnull().sum()















dataKNN = df1.copy()



from fancyimpute import KNN



data_cols = list(dataKNN)



dataKNN = dataKNN[['Salary', 'Age']]



dataKNN = pd.DataFrame(KNN(k=1).fit_transform(dataKNN))

df1.head(5)