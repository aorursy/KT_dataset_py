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
# Install the dependecies

import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

plt.style.use('bmh')
# Store the data into a data frame

df = pd.read_csv('../input/titanic/gender_submission.csv')

df.head()
# Get the number of trainig days

df.shape
#Visualize the close price data

plt.figure(figsize=(16,8))

plt.title('titanic')

plt.xlabel('Days')

plt.ylabel('PassengerId ')

plt.plot(df['PassengerId'])

plt.show()
# Get the close

df = df[['PassengerId']]

df.head()
# Create a variable to  predict 'x' days out into the future

future_days = 25

# Cereate a new column (target) shifted 'x' units/days up

df['prediction'] = df[['PassengerId']].shift(-future_days)

df.tail(4)
# Create the feature data set (x) and convert it to a numpy array and remove the list 'x' rows/days

x = np.array(df.drop(['prediction'], 1))[:-future_days]

print(x)
# Create the target data set (y) and convert it too numpy array and get all of the target values except the last rows/days

y = np.array(df.drop(['prediction'], 1))[:-future_days]

print(y)
# Split the data into 75% training adn 25% testing

x_train,x_test,y_train,y_test = train_test_split(x ,y, test_size = 0.25)
#create the models

# create the decisin model

tree = DecisionTreeRegressor().fit(x_train, y_train)

#create the linear regression model

lr =  LinearRegression().fit(x_train, y_train)
#get the last 'x' rows of the future data set

x_future = df.drop(['prediction'], 1)[:-future_days]

x_future = x_future.tail(future_days)

x_future = np.array(x_future)

x_future
#show the model tree prediction

tree_prediction = tree.predict(x_future)

print(tree_prediction)

print()

#show the model linear regession predction 

lr_prediction = lr.predict(x_future)

print(lr_prediction)
# visualize the data

predictions = tree_prediction



valid = df[x.shape[0]:]

valid ['predictions'] = predictions

plt.figure(figsize=(16,8))

plt.title('Model')

plt.xlabel('Days')

plt.ylabel(['PassengerId'])

plt.plot(df['PassengerId'])

plt.plot(valid[['PassengerId', 'predictions']])

plt.legend(['Orig', 'Val', 'Pred'])

plt.show()