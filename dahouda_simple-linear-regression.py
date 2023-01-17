# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing the dataset

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#Importing the dataset

ds = pd.read_csv("/kaggle/input/studentscore.csv")

ds.head()

X = ds.iloc[ : ,   : 1 ].values

Y = ds.iloc[ : , 1 ].values



#Fitting Simple Linear Regression Model to the training set

from sklearn.model_selection import train_test_split



#from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor = regressor.fit(X_train, Y_train)



#Predecting the Result

Y_pred = regressor.predict(X_test)



#Visualising the Training results

plt.scatter(X_train , Y_train, color = 'red')

plt.plot(X_train , regressor.predict(X_train), color ='green')



#Visualizing the test results

plt.scatter(X_test , Y_test, color = 'black')

plt.plot(X_test , regressor.predict(X_test), color ='blue')