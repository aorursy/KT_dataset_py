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
import matplotlib.pyplot as plt

import seaborn as sns
dataset=pd.read_csv('../input/bottle.csv')

dataset = dataset[:][:400]

dataset.head()
X=dataset.iloc[:,6].values.reshape(-1,1)

y=dataset.iloc[:,5].values.reshape(-1,1)
print(np.any(np.isnan(X)))

print(np.any(np.isnan(y)))
from sklearn.preprocessing import Imputer

imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

imputer=imputer.fit(X)

X=imputer.transform(X)

imputer=imputer.fit(y)

y=imputer.transform(y)
print(np.any(np.isnan(X)))

print(np.any(np.isnan(y)))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
plt.scatter(X_train,y_train)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('Sal vs Temp (Training set)')

plt.xlabel('Temp')

plt.ylabel('Sal')

plt.show()
plt.scatter(X_test, y_test, color = 'Blue')

plt.plot(X_test, regressor.predict(X_test), color = 'blue')

plt.title('Sal vs Temp (Testing set)')

plt.xlabel('Temp')

plt.ylabel('Sal')

plt.show()
acc=regressor.score(X_test,y_test)

acc