# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn.linear_model import LinearRegression



#Checking for nans

train_data  = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print(train_data.isnull().any())

print(test_data.isnull().any())



train_data = train_data[np.isfinite(train_data['y'])]



print(train_data.isnull().any())

print(test_data.isnull().any())



X_train = train_data.iloc[:, 0:1].values

y_train = train_data.iloc[:, 1].values



X_test = test_data.iloc[:, 0:1].values

y_test = test_data.iloc[:, 1].values

#fitting models

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor = regressor.fit(X_train, y_train)

regressor
#making predictions 

y_pred = regressor.predict(X_test)
#visualizing the trainset

import matplotlib.pyplot as plt

plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title("X vs y (Train set)")

plt.xlabel("x")

plt.ylabel("y")

plt.show()
#visualizing the testset

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color = 'red')

plt.plot(X_test, regressor.predict(X_test), color = 'blue')

plt.title("X vs y (Test set)")

plt.xlabel("x")

plt.ylabel("y")

plt.show()