# Importing libraries



import numpy as np

import pandas as pd

import sklearn.linear_model as lm

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.isnull().sum()
test.isnull().sum()
fig = plt.figure(figsize=(8, 5))

ax = fig.add_subplot(111)

ax = train.boxplot(column='Fare', by=['Embarked','Pclass'], ax=ax)

plt.axhline(y=80, color='green')

ax.set_title('', y=1.1)



train[train.Embarked.isnull()][['Fare', 'Pclass', 'Embarked']]



_ = train.set_value(train.Embarked.isnull(), 'Embarked', 'C')

fig = plt.figure(figsize=(8, 5))

ax = fig.add_subplot(111)

test[(test.Pclass==3)&(test.Embarked=='S')].Fare.hist(bins=100, ax=ax)

test[test.Fare.isnull()][['Pclass', 'Fare', 'Embarked']]

plt.xlabel('Fare')

plt.ylabel('Frequency')

plt.title('Histogram of Fare, Plcass 3 and Embarked S')



test[test.Fare.isnull()][['Pclass', 'Fare', 'Embarked']]



print ("The top 5 most common value of Fare")

test[(test.Pclass==3)&(test.Embarked=='S')].Fare.value_counts().head()



_ = test.set_value(test.Fare.isnull(), 'Fare', 8.05)



full = pd.concat([train, test], ignore_index=True)

_ = full.set_value(full.Cabin.isnull(), 'Cabin', 'U0')
full.head()
import re



names = full.Name.map(lambda x:  len(re.split(' ',x)))

full.set_value(full.index,'Names',names)
title = full.Name.map(lambda x: re.compile(', (.*?)\.').findall(x)[0])

title[title=='Mme'] = 'Mrs'

title[title.isin(['Ms','Mlle'])] = 'Miss'

title[title.isin(['Don', 'Jonkheer'])] = 'Sir'

title[title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

title[title.isin(['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev'])] = 'Officer'

_ = full.set_value(full.index, 'Title', title)

del title


deck = full[~full.Cabin.isnull()].Cabin.map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())

deck = pd.factorize(deck)[0]

_ = full.set_value(full.index, 'Deck', deck)

del deck
checker = re.compile("([0-9]+)")

def roomNum(x):

    nums = checker.search(x)

    if nums:

        return int(nums.group())+1

    else:

        return 1

rooms = full.Cabin.map(roomNum)

_ = full.set_value(full.index, 'Room', rooms)

del checker, roomNum

full['Room'] = full.Room/full.Room.sum()
full['Group_num'] = full.Parch + full.SibSp + 1
full['Group_size'] = pd.Series('M', index=full.index)

_ = full.set_value(full.Group_num>4, 'Group_size', 'L')

_ = full.set_value(full.Group_num==1, 'Group_size', 'S')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

full['NorFare'] = pd.Series(scaler.fit_transform(full.Fare.reshape(-1,1)).reshape(-1), index=full.index)
def setValue(col):

    _ = train.set_value(train.index, col, full[:891][col].values)

    _ = test.set_value(test.index, col, full[891:][col].values)



for col in ['Deck', 'Room', 'Group_size', 'Group_num', 'Names', 'Title']:

    setValue(col)
full.drop(labels=['PassengerId', 'Name', 'Cabin', 'Survived', 'Ticket', 'Fare'], axis=1, inplace=True)

full = pd.get_dummies(full, columns=['Embarked', 'Sex', 'Title', 'Group_size'])
from sklearn.model_selection import train_test_split

X = full[~full.Age.isnull()].drop('Age', axis=1)

y = full[~full.Age.isnull()].Age

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer



def get_model(estimator, parameters, X_train, y_train, scoring):  

    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)

    model.fit(X_train, y_train)

    return model.best_estimator_
import xgboost as xgb



XGB = xgb.XGBRegressor(max_depth=4, seed= 42)

scoring = make_scorer(mean_absolute_error, greater_is_better=False)

parameters = {'reg_alpha':np.linspace(0.1,1.0,5), 'reg_lambda': np.linspace(1.0,3.0,5)}

reg_xgb = get_model(XGB, parameters, X_train, y_train, scoring)

print (reg_xgb)
print ("Mean absolute error of test data: {}".format(mean_absolute_error(y_test, reg_xgb.predict(X_test))))
fig = plt.figure(figsize=(15, 6))

alpha = 0.5

full.Age.value_counts().plot(kind='density', color='#FA2379', label='Before', alpha=alpha)



pred = reg_xgb.predict(full[full.Age.isnull()].drop('Age', axis=1))

full.set_value(full.Age.isnull(), 'Age', pred)



full.Age.value_counts().plot(kind='density', label='After', alpha=alpha)

plt.xlabel('Age')

plt.title("What's the distribution of Age after predicting?" )

plt.legend(loc='best')

plt.grid()
full['NorAge'] = pd.Series(scaler.fit_transform(full.Age.reshape(-1,1)).reshape(-1), index=full.index)

full['NorNames'] = pd.Series(scaler.fit_transform(full.Names.reshape(-1,1)).reshape(-1), index=full.index)

full['Group_num'] = pd.Series(scaler.fit_transform(full.Group_num.reshape(-1,1)).reshape(-1), index=full.index)
for col in ['NorAge', 'NorFare', 'NorNames', 'Group_num']:

    setValue(col)
train.Sex = np.where(train.Sex=='female', 0, 1)

test.Sex = np.where(test.Sex=='female', 0, 1)
train.drop(labels=['PassengerId', 'Name', 'Names', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)

test.drop(labels=['Name', 'Names', 'Cabin', 'Ticket', 'Age', 'Fare'], axis=1, inplace=True)
train = pd.get_dummies(train, columns=['Embarked', 'Pclass', 'Title', 'Group_size'])

test = pd.get_dummies(test, columns=['Embarked', 'Pclass', 'Title', 'Group_size'])

test['Title_Sir'] = pd.Series(0, index=test.index)
test.head()