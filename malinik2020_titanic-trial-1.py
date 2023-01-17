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
titanic_file_path = '/kaggle/input/titanic/train.csv'
titanic_data=pd.read_csv(titanic_file_path)
print(titanic_data.columns)
y = titanic_data.Survived
print(y)
titanic_parameters=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Fare']
print(titanic_parameters)
X = titanic_data[titanic_parameters]
X.describe()
X.head()
#to clean up data and get rid of missing values

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

new_data = my_imputer.fit_transform(X)

print(new_data)
#machine learning model

from sklearn.tree import DecisionTreeRegressor

titanic_model = DecisionTreeRegressor(random_state=1)

titanic_model.fit(new_data,y)
#to predict with training data set

predictions = titanic_model.predict(new_data)
print(predictions)
#to predict with test data set

final_test = '/kaggle/input/titanic/test.csv'
final_titanic_data=pd.read_csv(titanic_file_path)
final_y = final_titanic_data.Survived
final_titanic_parameters=['PassengerId', 'Pclass', 'Age', 'SibSp', 'Fare']
final_X = final_titanic_data[final_titanic_parameters]
final_X.head()
final_new_data = my_imputer.fit_transform(final_X)

print(final_new_data)
final_titanic_model = DecisionTreeRegressor(random_state=1)

final_titanic_model.fit(final_new_data,final_y)
final_predictions = final_titanic_model.predict(final_new_data)
print(final_predictions)
print(final_X)