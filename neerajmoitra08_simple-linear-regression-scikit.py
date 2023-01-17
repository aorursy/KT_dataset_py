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
dataset_train= pd.read_csv('/kaggle/input/random-linear-regression/train.csv')

#dropping bad value from dataset  

dataset_train= dataset_train.drop(dataset_train[dataset_train['x'] >2500].index)

dataset_test= pd.read_csv('/kaggle/input/random-linear-regression/test.csv')

X_train = dataset_train.iloc[:,:-1].values

X_test = dataset_test.iloc[:,:-1].values

y_test = dataset_test.iloc[:,1:2].values

y_train = dataset_train.iloc[:,1:2].values
#taking care of missing data, only y column of train has missing data on line 215

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

y_train = my_imputer.fit_transform(y_train)



from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Pedicting the test set results

y_pred = regressor.predict(X_test)
#check accuracy

from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
#Visualize the training set results

import matplotlib.pyplot as plt

plt.scatter(X_train,y_train, color='red')

plt.plot(X_train,regressor.predict(X_train), color= 'blue' )

plt.xlabel('X')

plt.ylabel('Y')

plt.title('X vs Y')

plt.show()
#Visualize the test set results

plt.scatter(X_test,y_test, color='red')

plt.plot(X_train,regressor.predict(X_train), color = 'blue')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('X vs Y')

plt.show()


