# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import matplotlib.pyplot as plt

import seaborn as sns

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_absolute_error, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier

from xgboost import XGBClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import normalize

from sklearn.svm import SVC



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#retrieve data

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

# regroup data 

all_data = pd.concat([train, test], ignore_index=True)

#data info

train.info()

print("-" *60)

test.info()
# show summary statistics

train.describe()

test.describe()
plt.hist(train.Age)

plt.xlabel('Age')

train[['Age', 'Survived']].groupby('Age').count().plot.bar(figsize=(25,5))
# fill NaN with a random value close to median

all_data.Age = all_data.Age.fillna(random.randrange(all_data.Age.median()-5, all_data.Age.median()+5, 2))

fig,ax = plt.subplots(1,2,figsize=(12,5))

sns.barplot(x='Sex', y='Survived', data= train, ax=ax[0])

sns.countplot('Sex', hue='Survived', data= train, ax=ax[1])
fig, ax = plt.subplots(1, 3, figsize=(18,5))

sns.countplot(x='Pclass', data=train, ax=ax[0])

ax[0].set_title('Number of passengers by class')

sns.countplot(x='Pclass', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival counts per class')

sns.barplot(x='Pclass', y='Survived', data=train, ax=ax[2])

ax[2].set_title('Survival rate per class')
fig, ax = plt.subplots(1, 3, figsize=(15,5))

sns.countplot('Embarked', data=train, ax=ax[0])

ax[0].set_title('Number of passengers by port of embarcation ')

sns.countplot('Embarked', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival count by port')

sns.barplot('Embarked', 'Survived', data=train, ax=ax[2])

ax[2].set_title('Survival rate by port')
all_data.Embarked = all_data.Embarked.fillna('S')
fig, ax = plt.subplots(1, 2, figsize=(15,5))

sns.countplot('Parch', data=train, ax=ax[0])

ax[0].set_title('Nb of Passengers by Nb of children')

sns.countplot('Parch', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival count by Nb of children')
fig, ax = plt.subplots(1, 2, figsize=(15,5))

sns.countplot('SibSp', data=train, ax=ax[0])

ax[0].set_title('Nb of Passengers by Nb of siblings or/and spouse')

sns.countplot('SibSp', hue='Survived', data=train, ax=ax[1])

ax[1].set_title('Survival count by Nb of siblings or/and spouse')
plt.hist(train.Fare)

plt.title('Fare Distribution')

plt.xlabel('Fare')

plt.ylabel('Nb of passengers')
# handle missing values

all_data.Fare = all_data.Fare.fillna(all_data.Fare.median())

# know what columns still contain missing values

all_data.columns[all_data.isna().any()]
all_data['Family_size'] = all_data.Parch + all_data.SibSp

all_data['Parch_sq'] = all_data.Parch ** 2

all_data['SibSp_sq'] = all_data.SibSp ** 2

all_data['Family_size_sq'] = all_data.Family_size ** 2

all_data['Parch_sq2'] = all_data.Parch ** 4

all_data['SibSp_sq2'] = all_data.SibSp ** 4

all_data['Family_size_sq2'] = all_data.Family_size ** 4

all_data['Family_size_sq2'] = all_data.Family_size ** 10
all_data = all_data.join(pd.get_dummies(all_data[['Sex', 'Embarked', 'Cabin', 'Ticket']]))

all_data = all_data.drop(['Sex','Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
all_data['Age_cat']=0

all_data.loc[all_data['Age']<=16,'Age_cat']=0

all_data.loc[(all_data['Age']>16)&(all_data['Age']<=32),'Age_cat']=1

all_data.loc[(all_data['Age']>32)&(all_data['Age']<=48),'Age_cat']=2

all_data.loc[(all_data['Age']>48)&(all_data['Age']<=64),'Age_cat']=3

all_data.loc[all_data['Age']>64,'Age_cat']=4

all_data = all_data.drop(['Age'], axis=1)

all_data['Fare_cat']=0

all_data.loc[all_data['Fare']<=7.775,'Fare_cat']=0

all_data.loc[(all_data['Fare']>7.775)&(all_data['Fare']<=8.662),'Fare_cat']=1

all_data.loc[(all_data['Fare']>8.662)&(all_data['Fare']<=14.454),'Fare_cat']=2

all_data.loc[(all_data['Fare']>14.454)&(all_data['Fare']<=26.0),'Fare_cat']=3

all_data.loc[(all_data['Fare']>26.0)&(all_data['Fare']<=52.369),'Fare_cat']=4

all_data.loc[all_data['Fare']>52.369,'Fare_cat']=5

all_data = all_data.drop(['Fare'], axis=1)
new_train = all_data.iloc[:891]

new_test = all_data.iloc[891:]
feature_columns = list(new_train.columns)

feature_columns.remove('Survived')

feature_columns.remove('PassengerId')
X = new_train[feature_columns]

y = new_train.Survived

# split data

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size= 0.2, random_state=1)
model_rf = RandomForestClassifier(criterion='entropy', max_leaf_nodes=100, random_state=1)

model_rf = BaggingClassifier(base_estimator=model_rf, random_state=1)

model_rf.fit(train_X, train_y)

predictions = model_rf.predict(val_X)

accuracy = accuracy_score(predictions, val_y) * 100

print("Accuracy:", accuracy)
model_dt = DecisionTreeClassifier(criterion='gini', max_leaf_nodes=100, random_state=1)

model_dt = BaggingClassifier(base_estimator=model_dt, random_state=1)

model_dt.fit(train_X, train_y)

predictions = model_dt.predict(val_X)

accuracy = accuracy_score(predictions, val_y) * 100

print("Accuracy:", accuracy)
model_xgb = XGBClassifier(objective='binary:logistic', eta=0.01, max_depth=50, gamma=1)

#model_xgb = GridSearchCV(model_xgb,{'max_depth': [2, 4, 6], 'eta': [0.01, 0.1, 0.2]}, verbose=1)

model_xgb.fit(train_X, train_y)

predictions = model_xgb.predict(val_X)

accuracy = accuracy_score(predictions, val_y) * 100

print("Accuracy:", accuracy)
model_svc = SVC(C=10)

model_svc.fit(train_X, train_y)

predictions = model_svc.predict(val_X)

accuracy = accuracy_score(predictions, val_y) * 100

print("Accuracy:", accuracy)
model_nb = GaussianNB()

model_nb.fit(train_X, train_y)

predictions = model_nb.predict(val_X)

accuracy = accuracy_score(predictions, val_y) * 100

print("Accuracy:", accuracy)
# retrain model on all training data

model_rf.fit(X, y)

model_dt.fit(X, y)

model_xgb.fit(X, y)

model_svc.fit(X, y)

model_nb.fit(X, y)
# submit results

predictions1 = model_rf.predict(new_test[feature_columns])

predictions2 = model_dt.predict(new_test[feature_columns])

predictions3 = model_xgb.predict(new_test[feature_columns])

predictions4 = model_svc.predict(new_test[feature_columns])

predictions = (predictions1 + predictions2 + predictions3 + predictions4) / 4

predictions = [int(round(x)) for x in predictions]



submission = pd.DataFrame({'PassengerId':new_test['PassengerId'],'Survived':predictions})



#Visualize the first 20 rows

submission.head(20)
filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)