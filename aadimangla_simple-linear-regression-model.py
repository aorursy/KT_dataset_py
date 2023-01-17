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
#IMPORTING THE LIBRARIES

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



#IMPORTING THE TRAIN DATASET

dataset = pd.read_csv ('/kaggle/input/simple-linear-regression/train.csv') 

X=dataset.iloc[:, :-1].values

y=dataset.iloc[:, 1].values



#IMPORTING THE TEST DATASET

dataset_test = pd.read_csv ('/kaggle/input/random-linear-regression/test.csv') 

X_test=dataset_test.iloc[:, :-1].values

y_test=dataset_test.iloc[:, 1].values



#TAKING CARE OF MISSING DATA

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)

imputer=imputer.fit(X[:])

X[:]=imputer.transform(X[:])



from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)

imputer=imputer.fit(X_test[:])

X_test[:]=imputer.transform(X_test[:])



#FITTING THE LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X , y)

#pedicting the dataset

y_pred = regressor.predict(X_test)

#plotting the graph || visualizing training cases

plt.scatter(X,y, color='blue')

plt.plot(X,regressor.predict(X),color='red')

plt.title('salary vs experience')

plt.xlabel('years of experience')

plt.ylabel('salary')

plt.show()

#plotting the graph || visualizing test cases

plt.scatter(X_test,y_test, color='blue')

plt.plot(X,regressor.predict(X),color='red')

'''plt.title('salary vs experience')

plt.xlabel('years of experience')

plt.ylabel('salary')'''

plt.show()