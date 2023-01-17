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
np.random.seed(0)
#Creating variables for train_data and test_data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

#Describing train_data to see if there are any missing values.
train_data.describe()
#As we can see there are some missing values in 'Age', let's inspect the data
train_data['Age'].head(20)

#As suspected 'Age' has NaN values. To fix them we can substitute the ages with the mean age which is 29.69, roughly 30.
train_data['Age'] = train_data['Age'].fillna(30.0)
test_data['Age'] = test_data['Age'].fillna(30.0)

#There are also NaN values in test_data's 'Fare' column. We'll just replace it with the mean fare as well.
test_data['Fare'] = test_data['Fare'].fillna(30.0)

#For our model to work, we will also need to change the age to numeric values.
#We'll use a simple convention where female=1, male=0
train_data['Sex'] = train_data['Sex'].replace('male',0)
train_data['Sex'] = train_data['Sex'].replace('female',1)
test_data['Sex'] = test_data['Sex'].replace('male',0)
test_data['Sex'] = test_data['Sex'].replace('female',1)
        
#Let's describe the data now
test_data.describe()
#train_data.describe()
#Looks like we've fixed the 'Age' column and we're back to 891 values. The mean is also not as changed as it could have been.

#Let's print all columns to find the prediction target
train_data.columns
#It's clear we're looking at the 'Survived' column when we're looking to make predictions.

y = train_data.Survived
#Let's define X

features = ['Pclass','Sex','Age','SibSp','Parch','Fare']

X = train_data[features]
#Let's review X before we proceed

print(X.describe())
print("\n")
print(X.head())
from sklearn.model_selection import train_test_split
#Splitting the data into train and validation

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
from sklearn.tree import DecisionTreeRegressor

#setting random_state=1 for reproducibility
titanic_model = DecisionTreeRegressor(random_state=1)

#fitting the model now
titanic_model.fit(train_X,train_y)
#We're just testing how well fitted the model is here.
titanic_preds = titanic_model.predict(val_X)
from sklearn.metrics import mean_absolute_error

#Let's calculate the MAE

titanic_mae = mean_absolute_error(titanic_preds, val_y)

print(titanic_mae)

#making an X function

final_X = test_data[features]
final_predictions = titanic_model.predict(final_X)
final_predictions = np.round(final_predictions)
final_predictions = final_predictions.astype(int)
#creating the object
submission = pd.DataFrame({
    'PassengerId':test_data['PassengerId'],
    'Survived':final_predictions
})
print(submission.head())
submission.to_csv('titanic_submission.csv', index = False)