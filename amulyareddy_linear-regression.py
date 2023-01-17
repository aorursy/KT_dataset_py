# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/random-linear-regression/test.csv')

X_test, y_test = test[['x']],test[['y']]
print(X_test.head())

print(y_test.head())
train = pd.read_csv('../input/random-linear-regression/train.csv')

train = train.dropna()

X_train, y_train = train[['x']],train[['y']]
from sklearn.linear_model import LinearRegression

model = LinearRegression()
X_train = X_train.astype(float)

y_train = y_train.astype(float)

print(X_train.head())

print(y_train.head())
avg_x = np.sum(X_train)/len(X_train)

avg_y = np.sum(y_train)/len(y_train)

print(avg_x,avg_y)
model.fit(X_train,y_train)
# X_test = X_test.astype(float)

# avg_x = np.sum(X_test)/len(X_test)

# X_test = X_test.fillna(avg_x)
predicted = model.predict(X_test.dropna())
plt.plot(X_test,y_test,'o')

plt.plot(X_test,predicted,'r')