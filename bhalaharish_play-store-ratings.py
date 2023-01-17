# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
ds = pd.read_csv('../input/google-play-store-apps/googleplaystore.csv')
ds.head(10)
type(ds)
ds.shape
ds.describe()
ds.boxplot()
ds.hist()
ds.info()
ds.isnull()
ds.isnull().sum()
ds[ds.Rating>5]
ds.drop([10472], inplace = True)  #Inplace = True removes the row from the dataset
ds[10470:10475]
ds.boxplot()
ds.hist()
threshold = len(ds)*0.1

threshold
ds.dropna(thresh = threshold, axis = 1, inplace = True)  # axis = 0 ---> row and 1 ---> column
print(ds.isnull().sum())

ds.shape
# Defining a function to fill missing Ratings with Median (Right Skewed)

def fill(series):

    return series.fillna(series.median())

    
ds.Rating = ds['Rating'].transform(fill)
ds.isnull().sum()
#Modes of Categorical Values

print(ds['Type'].mode())

print(ds['Current Ver'].mode())

print(ds['Android Ver'].mode())
ds['Type'].fillna(str(ds['Type'].mode().values[0]), inplace = True)

ds['Current Ver'].fillna(str(ds['Current Ver'].mode().values[0]), inplace = True)  #values[0] is added so that in case there are two modes, the value of the first one is used

ds['Android Ver'].fillna(str(ds['Android Ver'].mode().values[0]), inplace = True)
ds.isnull().sum()
# Converting Price, Reviews and Ratings into Numerical Data

ds['Price'] = ds['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

ds['Price'] = ds['Price'].apply(lambda x: float(x))

ds['Reviews'] = pd.to_numeric(ds['Reviews'], errors = 'coerce')
ds['Installs'] = ds['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))

ds['Installs'] = ds['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))

ds['Installs'] = ds['Installs'].apply(lambda x: float(x))
ds.tail(10)
ds.describe()
grp = ds.groupby('Category')

x = grp['Rating'].agg(np.mean)

y = grp['Price'].agg(np.sum)

z = grp['Reviews'].agg(np.mean)

print(x)

print(y)

print(z)
plt.figure(figsize=(16,5))

plt.plot(x, '-.r')   #o seperates the values instead of a continuous plot , b gives blue colour

plt.xticks(rotation = 90)

plt.title('Category wise Rating')

plt.show
plt.figure(figsize=(16,5))

plt.plot(y, '-.b')   

plt.xticks(rotation = 90)

plt.title('Category wise Total Price')

plt.show
plt.figure(figsize=(16,5))

plt.plot(z, '-.g')   

plt.xticks(rotation = 90)

plt.title('Category wise Reviews')

plt.show