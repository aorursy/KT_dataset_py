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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

#this is used to impute the mean and fill missing values to the missing column

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier





survived_female = train_data.loc[train_data.Sex=='female']['Survived']

survived_female_percent = sum(survived_female )/len(survived_female )

survived_female_percent
survived_male = train_data.loc[train_data.Sex=='male']['Survived']

survived_male_percent = sum(survived_male )/len(survived_male )

survived_male_percent
#create target object and call it y

y=train_data.Survived

#creat X

features= ['Pclass', 'Sex','Age','SibSp', 'Parch']



X = train_data[features]



#Changing 'Sex' Categorical data to Binary Columns

one_hot_encoded_X =pd.get_dummies(X)

#calling SimpleImpute function to fill the blank values with mean or any statisticl data

my_imputer= SimpleImputer()

train_data_with_imputed = my_imputer.fit_transform(one_hot_encoded_X)



train_data_with_imputed

#Fit the model

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(train_data_with_imputed ,y)



#Tackling test Data

#In the first line we have converted the Sex categorical data in to binary form

n_test_data=pd.get_dummies(test_data[features])

#Now inputing the mean and filling the null age values with mean

test_data_with_imputed= my_imputer.fit_transform(n_test_data)

test_data_with_imputed

prediction=model.predict(test_data_with_imputed)

prediction

output=pd.DataFrame({'PassengerID': test_data.PassengerId,'Survived':prediction })

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")