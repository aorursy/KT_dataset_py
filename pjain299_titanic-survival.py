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
train_data = pd.read_csv("../input/train.csv")

train_data.columns
test_data = pd.read_csv("../input/test.csv")

test_data.columns
gs_data = pd.read_csv("../input/gender_submission.csv")

gs_data.columns
train_data.head()
train_data.info()

print('_'*40)

test_data.info()
train_data.describe() 
train_data.describe(include=['O'])
train_data.isnull().sum()
train_data.groupby('Embarked').Pclass.value_counts()
train_test_data = [train_data,test_data] #Combining train and test dataset
for dataset in train_test_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
for dataset in train_test_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    
for dataset in train_test_data:

    dataset.loc[ dataset['Age'] <= 8, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 8)  & (dataset['Age'] <= 16), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 4

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 5



train_data.head()
for dataset in train_test_data:

    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())



for dataset in train_test_data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)
for dataset in train_test_data:

    dataset['FamilySize'] = dataset['SibSp'] +  dataset['Parch'] + 1
features_drop = ['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']

train_data = train_data.drop(features_drop, axis=1)

test_data = test_data.drop(features_drop, axis=1)

train_data = train_data.drop(['PassengerId'], axis=1)
train_data.head()
test_data.head()
X_train = train_data.drop('Survived', axis=1)

y_train = train_data['Survived']

X_test = test_data.drop("PassengerId", axis=1).copy()
from xgboost import XGBRegressor



model = XGBRegressor()

model.fit(X_train,y_train,verbose=False)

score = round(model.score(X_train,y_train) * 100, 2)

print (score)
from sklearn.ensemble import RandomForestClassifier



model =  RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

y_test = model.predict(X_test)

score = round(model.score(X_train,y_train) * 100, 2)

print (score)
submission = pd.DataFrame({

        "PassengerId": test_data["PassengerId"],

        "Survived": y_test

    })

submission.to_csv('submission.csv', index=False)