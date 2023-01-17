# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as seabornInstance

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

dataset = pd.read_csv("../input/akpam-vs-ips/akpam.csv")

dataset.shape
dataset.plot(x='Akpam', y='ips', style='o')

plt.title('AKPAM vs IPS')

plt.xlabel('Akpam')

plt.ylabel('ips')

plt.show()
X = dataset['ips'].values.reshape(-1,1)

y = dataset['Akpam'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()

regressor.fit(X_train, y_train) #training the algorithm

print(regressor.intercept_)

print(regressor.coef_)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')

plt.plot(X_test, y_pred, color='red', linewidth=2)

plt.show()
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))