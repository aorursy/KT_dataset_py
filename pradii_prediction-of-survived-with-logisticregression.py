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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sea

%matplotlib inline

from sklearn import metrics

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
train_data.head(4)
train_data.describe()

train_data.isna().sum()
train_data.isna().sum()
def dividing_map(x):

    if(x <= 12):

        return "Kid"

    elif(x >= 13 and x <= 19):

        return "Teen"

    elif(x >= 19 and x < 60):

        return "Adult"

    else:

        return "Senior"

train_data['Age'] = train_data['Age'].apply(lambda x: dividing_map(x))
train_data["Age"]
train_data['Cabin'].fillna('U', inplace=True)

train_data['Cabin'] = train_data['Cabin'].apply(lambda x: x[0])



train_data['Embarked'].fillna('S', inplace=True)
train_data.isna().sum()
train_data['Fare'] = pd.cut(train_data['Fare'],bins = [-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6]) 
Survived_in_pclass = sea.countplot(x = 'Survived', hue = 'Pclass', data = train_data)

plt.title("Survived: Frequency of Pclass ")

plt.show('Survived_in_class')
Survived_div_gender = sea.countplot(x = 'Sex', hue = 'Survived', data = train_data)

plt.title("Sex : Frequency of survived")

plt.show('Survived_div_gender')
fig, ax = plt.subplots(1,2,figsize=(10,5))

sea.countplot(train_data['Age'], data=train_data, ax=ax[0])

sea.countplot(train_data['Age'], hue='Survived', data=train_data, ax=ax[1])

ax[0].title.set_text("Counts of each age group")

ax[1].title.set_text("Survived: Age Group")
leg = LabelEncoder()

train_data['Sex'] = leg.fit_transform(train_data['Sex'])

train_data['Cabin'] = leg.fit_transform(train_data['Cabin'])

train_data['Embarked'] = leg.fit_transform(train_data['Embarked'])

train_data['Age'] = leg.fit_transform(train_data['Age'])
train_data.head()
plt.figure(figsize = (8,6))

cor = train_data.corr()

sea.heatmap(cor, annot = True)
train_data.drop(['Name'], axis = 1, inplace = True)
train_data.drop(['Ticket'], axis = 1, inplace = True)
y = train_data['Survived']

X = train_data.drop(['Survived'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.65, random_state = 3)
logistic = LogisticRegression()

model_log = logistic.fit(X_train, y_train)
predictions = model_log.predict(X_test)
def cleaning():

    test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)

    test_data['Age'] = test_data['Age'].apply(lambda x: dividing_map(x))

    test_data['Cabin'].fillna('U', inplace=True)

    test_data['Cabin'] = test_data['Cabin'].apply(lambda x: x[0])

    test_data['Embarked'].fillna('S', inplace=True)

    test_data['Sex'] = leg.fit_transform(test_data['Sex'])

    test_data['Cabin'] = leg.fit_transform(test_data['Cabin'])

    test_data['Embarked'] = leg.fit_transform(test_data['Embarked'])

    test_data['Age'] = leg.fit_transform(test_data['Age'])

    test_data.drop(['Name', 'Ticket'], axis = 1, inplace = True)

    fare_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)

    test_data['Fare'] = fare_imputer.fit_transform(test_data[['Fare']])

    test_data['Fare'] = pd.cut(test_data['Fare'],bins = [-1, 7, 11, 15, 22, 40, 520], labels=[1, 2, 3, 4, 5, 6]) 



cleaning()

preds = model_log.predict(test_data)
output = pd.DataFrame({'PassengerID': test_data['PassengerId'], 'Survived': preds})

output.to_csv("submission.csv", index=False)