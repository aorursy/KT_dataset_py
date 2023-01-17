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
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head();
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head();
train_data["Age"] = train_data["Age"].fillna(-0.5)

test_data["Age"] = test_data["Age"].fillna(-0.5)
train_data["CabinBool"] = (train_data["Cabin"].notnull().astype('int'))

test_data["CabinBool"] = (test_data["Cabin"].notnull().astype('int'))
train_data = train_data.drop(['Cabin'], axis = 1)

test_data = test_data.drop(['Cabin'], axis = 1)
train_data = train_data.drop(['Ticket'], axis = 1)

test_data = test_data.drop(['Ticket'], axis = 1)
train_data['Embarked'].unique()

print(train_data['Embarked'].unique())
train_data.loc[train_data['Embarked'].isnull(),'Embarked'] = 'C'
southampton = train_data[train_data["Embarked"] == "S"].shape[0]

cherbourg = train_data[train_data["Embarked"] == "C"].shape[0]

queenstown = train_data[train_data["Embarked"] == "Q"].shape[0]
train_data_data = train_data.fillna({"Embarked": "S"})
combine = [train_data, test_data]



for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train_data.head();
test_data.describe(include="all");
mr_age = train_data[train_data["Title"] == 1]["Age"].mode()

miss_age = train_data[train_data["Title"] == 2]["Age"].mode()

mrs_age = train_data[train_data["Title"] == 3]["Age"].mode()

master_age = train_data[train_data["Title"] == 4]["Age"].mode()

royal_age = train_data[train_data["Title"] == 5]["Age"].mode()

rare_age = train_data[train_data["Title"] == 6]["Age"].mode()



age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



for x in range(len(train_data["Age"])):

    if train_data["Age"][x] == "Unknown":

        train_data["Age"][x] = age_title_mapping[train_data["Title"][x]]

        

for x in range(len(test_data["Age"])):

    if test_data["Age"][x] == "Unknown":

        test_data["Age"][x] = age_title_mapping[test_data["Title"][x]]
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train_data['Age'] = train_data['Age'].map(age_mapping)

test_data['Age'] = test_data['Age'].map(age_mapping)



train_data.head()



train_data = train_data.drop(['Age'], axis = 1)

test_data = test_data.drop(['Age'], axis = 1)
train_data = train_data.drop(['Name'], axis = 1)

test_data = test_data.drop(['Name'], axis = 1)
sex_mapping = {"male": 0, "female": 1}

train_data['Sex'] = train_data['Sex'].map(sex_mapping)

test_data['Sex'] = test_data['Sex'].map(sex_mapping)



train_data.head();
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train_data['Embarked'] = train_data['Embarked'].map(embarked_mapping)

test_data['Embarked'] = test_data['Embarked'].map(embarked_mapping)



train_data.head();
for x in range(len(test_data["Fare"])):

    if pd.isnull(test_data["Fare"][x]):

        pclass = test_data["Pclass"][x]

        test_data["Fare"][x] = round(train_data[train_data["Pclass"] == pclass]["Fare"].mean(), 4)

        

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4, labels = [1, 2, 3, 4])

test_data['FareBand'] = pd.qcut(test_data['Fare'], 4, labels = [1, 2, 3, 4])



train_data = train_data.drop(['Fare'], axis = 1)

test_data = test_data.drop(['Fare'], axis = 1)
from sklearn.model_selection import train_test_split



predictors = train_data.drop(['Survived', 'PassengerId'], axis=1)

target = train_data["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val), 2)

print(acc_randomforest)
y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)
output = pd.DataFrame({ 'PassengerId' : test_data.PassengerId, 'Survived': predictions })

output.to_csv('my_submission10.csv', index=False)

print("Your submission was successfully saved!")