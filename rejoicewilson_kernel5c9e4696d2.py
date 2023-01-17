# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot  as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
car = pd.read_csv(r'/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')

car.head()
X = car.iloc[:,:-1].values

print (X)
y = car.iloc[:,25].values

print(y)
sns.heatmap(car.corr())
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()

X[:,2] = labelencoder.fit_transform(X[:,2])

X[:,3] = labelencoder.fit_transform(X[:,3])

X[:,4] = labelencoder.fit_transform(X[:,4])

X[:,5] = labelencoder.fit_transform(X[:,5])

X[:,6] = labelencoder.fit_transform(X[:,6])

X[:,7] = labelencoder.fit_transform(X[:,7])

X[:,8] = labelencoder.fit_transform(X[:,8])

X[:,14] = labelencoder.fit_transform(X[:,14])

X[:,15] = labelencoder.fit_transform(X[:,15])

X[:,17] = labelencoder.fit_transform(X[:,17])

X
df1=pd.DataFrame(X)
df1.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)
y_pred = lin_reg.predict(X_test)
y_pred
X_test
print(lin_reg.coef_)
print(lin_reg.intercept_)
from sklearn.metrics import r2_score

r2_score(y_test,y_pred)