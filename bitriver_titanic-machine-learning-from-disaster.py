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
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
gender_submission.describe()
train.describe(include="all")
train.isna().sum()
test.isna().sum()
#Find mean of age across Train and Test



age = pd.concat([train.Age, test.Age], axis=0)

print(age.median())

age.describe()

train.Age = train.Age.fillna(age.median())

test.Age = test.Age.fillna(age.median())
train.Embarked.value_counts()
train.Embarked = train.Embarked.fillna('S')

train
test
y_test = gender_submission[gender_submission["PassengerId"] == test["PassengerId"]]["Survived"]
y_test
train = train.drop(columns=['Fare', 'Name', 'Parch', 'PassengerId', 'SibSp', 'Ticket','Cabin'])

x_test  = test.drop(columns=['Fare', 'Name', 'Parch', 'PassengerId', 'SibSp', 'Ticket','Cabin'])
train.Sex  =  train.Sex.apply(lambda x: 0 if x == 'female' else 1)

x_test.Sex = x_test.Sex.apply(lambda x: 0 if x == 'female' else 1)
train.isna().sum()
gender_submission
from sklearn import preprocessing

lbe = preprocessing.LabelEncoder()

train.Embarked = lbe.fit_transform(train.Embarked)

x_test.Embarked = lbe.fit_transform(x_test.Embarked)

from sklearn.metrics import accuracy_score

x_train = train[['Pclass','Sex','Age', 'Embarked']]

y_train = train[['Survived']]





from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier



models = [RandomForestClassifier(), LogisticRegression(), GradientBoostingClassifier()]



predictions = []

max_s = 0

max_idx = -1

for i in range(len(models)):

    models[i].fit(x_train, y_train)

    pred = models[i].predict(x_test)

    predictions.append(pred)

    score = round(accuracy_score(pred, y_test) * 100, 2)

    if max_s < score:

        max_idx = i

        max_s = score

    print (score)

    



    

output = pd.DataFrame({ 'PassengerId' : test.PassengerId, 'Survived': predictions[max_idx] })

output.to_csv("submission.csv", index=False)