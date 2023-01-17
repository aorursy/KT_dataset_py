import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline


data = pd.read_csv('/kaggle/input/diamonds/diamonds.csv')

data.head()
data.size
data.shape
data.info()
data.describe()
data.corr()
data.isnull().sum()
data.drop(['Unnamed: 0'],inplace = True, axis = 1)

data.head()
print('Number of rows with x = 0 are {}'.format((data.x==0).sum()))

print('Number of rows with x = 0 are {}'.format((data.y==0).sum()))

print('Number of rows with x = 0 are {}'.format((data.z==0).sum()))
data.x = data.x.replace(0,np.NaN)

data.y = data.y.replace(0,np.NaN)

data.z = data.z.replace(0,np.NaN)



print("Number of rows with x = 0 are {}".format((data.x==0).sum()))

print("Number of rows with y = 0 are {}".format((data.y==0).sum()))

print("Number of rows with z = 0 are {}".format((data.z==0).sum()))
data.isna().sum()
data.dropna(inplace=True)

data.isna().sum()
sns.catplot(data=data, x='cut', kind = "count")
sns.catplot(data=data,x='clarity', y= 'price', kind = 'bar')
sns.catplot(data=data, x='color', y = 'price', kind = "bar")
sns.catplot(data=data, x='clarity', y = 'price', kind = "box")
print(sns.catplot(data=data, x='cut', y = 'price', kind = "box"))