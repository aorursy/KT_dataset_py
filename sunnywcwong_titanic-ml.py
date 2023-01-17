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
train_filepath = '../input/train.csv'

titanic_data = pd.read_csv(train_filepath)



print(titanic_data.describe())

print(titanic_data.columns)

titanic_data.head()
titanic_features = ['Fare', 'Age']

titanic_data = titanic_data.dropna(subset=titanic_features)

X = titanic_data[titanic_features]

y = titanic_data.Survived



print(X.describe())

print(X.head())

print(y.head())
from sklearn.tree import DecisionTreeRegressor



#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

titanic_model = DecisionTreeRegressor(random_state=1)



# Fit the model

titanic_model.fit(X,y)



predictions = np.round(titanic_model.predict(X))

print(predictions)
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y, predictions)

test_filepath = '../input/test.csv'

test_data = pd.read_csv(test_filepath)



test_data = test_data.dropna(subset=titanic_features)

X = test_data[titanic_features]



predictions = np.round(titanic_model.predict(X))

print(predictions)

titanic_model.fit(X,predictions)

#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':predictions})



#Visualize the first 5 rows

submission.head()
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)