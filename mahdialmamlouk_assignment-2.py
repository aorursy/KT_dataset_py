# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('../input/kc-house-data/kc_house_data.csv')

data
data = data.dropna() 

data

pd.DataFrame(data.isna().sum()).T
data = data.drop_duplicates()

print(data)
data
print(data.columns)

data.drop(['id','date', 'zipcode'], axis=1, inplace=True)

data 
# generate related variables

from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from matplotlib import pyplot



corr = data.corr()

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(corr,cmap='coolwarm', vmin=-0.8, vmax=0.8)

fig.colorbar(cax)

ticks = np.arange(0,len(data.columns),1)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(data.columns)

ax.set_yticklabels(data.columns)

plt.show()

#Spilt my dataset into training and testing sets :

X = data

y = data.price

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.30)

print("test dataset size after spliting is : ",X_test.shape)
m0 = open('trainfile.csv', 'w')

arr1 = str(X_train)

m0.write(arr1)

m0.close

m1 = open('testfile.csv', 'w')

arr2 = str(X_test)

m1.write(arr2)

m1.close