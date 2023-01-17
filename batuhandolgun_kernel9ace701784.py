# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ARCH4450_Final_BatuhanDolgun_PnarEngr_MerveKoyiit_MehmetTakran.csv')

data.head(5)
data.info()
data.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(data)
X= data[['FLOOR','WALL','COLOUR']]

X.head()
data.tail()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(5,5))

sns.heatmap(data.corr(), annot=True)
y= data['COLOUR']

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100 )
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
from sklearn.linear_model import LinearRegression 

lr = LinearRegression()#Creating a LinearRegression object

lr.fit(X_train, y_train)