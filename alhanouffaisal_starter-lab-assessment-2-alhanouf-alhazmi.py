# alhanouf faisal alhazmi

# 1675246

# lab assessment2 
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



dataavrege=df1.copy()

dataknn=df1.copy()
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





# data_file.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/data_file.csv', delimiter=',')

df1.dataframeName = 'data_file.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
#Q2

list(df1)
#Q3

df1.dtypes
#Q4

df1.get_dtype_counts() 
#Q5 change salary to float 



change = {'Salary' : float}

df1 = df1.astype(change)

print(df1.dtypes)

#Q6

df1.describe()
#Q7

print(pd.isnull(df1).sum())
#Q8





mean_value=dataavrege['Age'].mean()

dataavrege['Age']=dataavrege['Age'].fillna(mean_value)



print(dataavrege)#8

print(pd.isnull(dataavrege).sum())
#Q9



from fancyimpute import KNN

data_cols = list(df1)

dataknn = df1[['Age', 'Salary']]

dataknn = pd.DataFrame(KNN(k=1).fit_transform(dataknn))



 
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)