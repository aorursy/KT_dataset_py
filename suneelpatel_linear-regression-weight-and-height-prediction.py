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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/weight_and_height.csv")

data.shape
data.head()
print("Total Male:",(data['Gender']=='Male').sum())
sns.countplot(data['Gender'])
data = data.drop('Gender', axis=1)
data.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = data.iloc[:, :-1].values

y = data.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.30, random_state = 0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
pd.DataFrame({'Actual': y_test, 'Predict': y_pred}).head()
plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, lr.predict(X_train), color = 'blue')

plt.title('Height vs Weight (Training set)')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_train, lr.predict(X_train), color = 'blue')

plt.title('Height vs Weight (Test set)')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
import sklearn

mse = sklearn.metrics.mean_squared_error(y_test, y_pred)

print(mse)