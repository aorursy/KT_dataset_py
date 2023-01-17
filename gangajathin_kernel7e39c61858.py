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
traindata = pd.read_csv("/kaggle/input/random-linear-regression/train.csv")

testdata = pd.read_csv("/kaggle/input/random-linear-regression/test.csv")

traindata = traindata.drop(traindata[traindata['x'] > 2500].index)
X_train = traindata.iloc[:,:-1].values

Y_train = traindata.iloc[:,1:2].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean',verbose = 0)

Y_train[:,:] = imputer.fit_transform(Y_train[:,:])

X_test = testdata.iloc[:,:-1].values

Y_test = testdata.iloc[:,1:2].values

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)
Y_test

y_pred

from sklearn.metrics import r2_score

r2_score(Y_test,y_pred)
import matplotlib.pyplot as plt

plt.scatter(X_train, Y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('SLR')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()



# Visualising the Test set results

plt.scatter(X_test, Y_test, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')

plt.title('SLR')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()