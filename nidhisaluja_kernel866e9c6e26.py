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
#Importing Libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

# Importing the dataset

dataset = pd.read_csv('../input/heights-and-weights/data.csv')

dataset.head()
#checking null values

dataset.isnull().sum()
#Separating Input and Output Data

X = dataset.iloc[:,0:1].values

y = dataset.iloc[:,1].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

#Fitting Linear Regression Model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Predicting and comparing the Test set results

y_pred = regressor.predict(X_test)

print(y_test,y_pred)
#Graph for Train set

%matplotlib inline

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()
#Graph for Test Set

plt.scatter(X_test,y_test,color='red')

plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.xlabel('Height')

plt.ylabel('Weight')

plt.show()