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
import matplotlib.pyplot as plt

%matplotlib inline
myfile=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

myfile
myfile.shape
myfile.dtypes
myfile.describe()
myfile.boxplot()
myfile.hist()
myfile.info()
myfile[myfile.Rating>5]
myfile.drop([10472],inplace=True)

myfile[10470:10476]
myfile.hist()
myfile.Rating=myfile.Rating.fillna(myfile.Rating.median())
myfile.isnull().sum()
myfile.Type= myfile.Type.fillna(myfile.Type.mode().values[0])

myfile['Current Ver']= myfile['Current Ver'].fillna(myfile['Current Ver'].mode().values[0])

myfile['Android Ver']= myfile['Current Ver'].fillna(myfile['Android Ver'].mode().values[0])
myfile.isnull().sum()
myfile.dtypes
myfile.Reviews= pd.to_numeric(myfile.Reviews,errors='coerce')
myfile.Installs=myfile.Installs.apply(lambda x: str(x).replace('+','') if '+' in str(x) else str(x))

myfile.Installs=myfile.Installs.apply(lambda x: str(x).replace(',','') if ',' in str(x) else str(x))

myfile.Installs=pd.to_numeric(myfile.Installs)

myfile.Price=myfile.Price.apply(lambda x: str(x).replace('$','') if '$' in str(x) else str(x))

myfile.Price=myfile.Price.astype(float)
myfile.dtypes
myfile.describe()
grp_myfile=myfile.groupby('Category')

a=grp_myfile.Rating.mean()

b=grp_myfile.Reviews.mean()

c=grp_myfile.Price.mean()

print(a)

print(b)

print(c)







plt.figure(figsize=(20,10))

plt.xticks(rotation=90)

plt.plot(a)
plt.figure(figsize=(20,10))

plt.xticks(rotation=90)

plt.plot(b)
plt.figure(figsize=(20,10))

plt.xticks(rotation=90)

plt.plot(c)