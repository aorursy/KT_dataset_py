# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

labels = train.Survived

test_id = test.PassengerId

train.head()
train.describe()
print('Total on board: {}, Total Survived: {}, Survival Rate: {:4.2}'.format(len(train), 

                                                                         train.Survived.value_counts().loc[1],

                                                                         train.Survived.mean()))
all_data = pd.concat([train.drop('Survived', axis=1), test])
all_data.isnull().sum()
null_ages = all_data[all_data.Age.isnull()].groupby(['Pclass', 'Sex', 'Parch', 'SibSp']).size()

null_ages
age_analysis = all_data.groupby(['Pclass', 'Sex', 'Parch', 'SibSp']).Age.agg(['mean','count'])

age_analysis[age_analysis['count'] < 1] # Unique groups with NaN age
all_data[(all_data.SibSp == 8) | (all_data.Parch == 9)]
all_data['Age'].loc[159] = 14.5

all_data['Age'].loc[342] = 60

all_data['Age'].loc[365] = 55
all_data.Age.fillna(all_data.groupby(['Pclass', 'Sex', 'Parch', 'SibSp']).Age.transform('mean'), inplace=True)
all_data[all_data.Age.isnull()]
all_data.Age.fillna(train.groupby(['Pclass', 'Sex']).Age.transform('mean'), inplace=True)
all_data[all_data.Embarked.isnull()]
all_data[(all_data.Pclass == 1) & (all_data.Sex == 'female')].groupby(['Embarked', 'Sex', 'Pclass']).size()
all_data.Embarked.fillna('C', inplace=True)
all_data[all_data.Fare.isnull()]
all_data.Fare.fillna(all_data.groupby(['Pclass']).Fare.transform('mean'), inplace=True)
plt.figure(figsize=(8,4))

plt.subplot(121)

sns.barplot(train.SibSp, train.Survived)

plt.subplot(122)

sns.barplot(train.Parch, train.Survived)

plt.subplots_adjust(wspace=0.5)

plt.show()
print(train.groupby(['SibSp']).size(), train.groupby(['Parch']).size())
all_data['Family'] = all_data.SibSp + all_data.Parch

all_data['Common_Tickets'] = all_data.groupby(['Ticket']).PassengerId.transform(np.size) - 1

all_data['Friends'] = all_data.Common_Tickets - all_data.Family 

all_data['Alone'] = ((all_data.Family < 1) & (all_data.Friends < 1)).astype(int)

all_data['Sex'] = all_data.Sex.apply(lambda x: 0 if x=='female' else 1)

all_data['Embarked_C'] = all_data.Embarked.apply(lambda x: 1 if x=='C' else 0)

all_data['Embarked_Q'] = all_data.Embarked.apply(lambda x: 1 if x=='Q' else 0)

all_data['Embarked_S'] = all_data.Embarked.apply(lambda x: 1 if x=='S' else 0)



all_data.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)

train = all_data.iloc[:len(train)]

test = all_data.iloc[len(train):]



pd.concat([train, labels], axis=1).corr().style.background_gradient(cmap='coolwarm') # Plot the correlation matrix
# Scoring..

features = ['Sex', 'Age', 'Pclass', 'Fare', 'Family', 'Alone', 'Embarked_C', 'Embarked_Q', 'Embarked_S']

selected_train = train[features]

x_train, x_test, y_train, y_test = train_test_split(selected_train, labels, train_size=0.8, test_size=0.2)



# Scale the feature data so it has mean = 0 and standard deviation = 1

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)



# Create and train the model

model = LogisticRegression()

model.fit(x_train, y_train)



# Score the model on the train data

print(model.score(x_train, y_train))



# Score the model on the test data

print(model.score(x_test, y_test))
selected_test = test[features]

scaler = StandardScaler()

train = scaler.fit_transform(selected_train)

test = scaler.transform(test[features])

model = LogisticRegression()

model.fit(train, labels)

submission = pd.DataFrame({"PassengerId": test_id, "Survived": model.predict(test)})

submission.to_csv('submission.csv', index=False)