import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df_train = pd.read_csv("../input/titanic/train.csv")

df_test = pd.read_csv("../input/titanic/test.csv")

td = pd.concat([df_train, df_test], ignore_index=True, sort = False)

td.head()

numb=td['PassengerId']
td.info()
#проводим факторизацию признаков

td['Name'] = pd.factorize(td['Name'])[0]  

td['Cabin'] = pd.factorize(td['Cabin'])[0]  

td['Ticket'] = pd.factorize(td['Ticket'])[0]  

td.drop(['Name', 'Cabin', 'Ticket','PassengerId'], axis=1, inplace=False)
d=549/(549+342)

print (d)
td['Sex'].describe()
td['Fare'].describe()
td['Embarked'].describe()
td['Survived'].describe()
#дополняем пропущенными значениями

td['Sex'].fillna('male', inplace=True)

td ['Embarked'].fillna('Embarked', inplace=True)

td['Age'].fillna(td['Age'].median(), inplace=True)

td['Fare'].fillna(td['Fare'].median(), inplace=True)
#кодируем признаки при помощи разбиения на категории

td = pd.concat([td,

               pd.get_dummies(td['Sex'], prefix="Sex"),

               pd.get_dummies(td['Age'], prefix="Age"),

               pd.get_dummies(td['Fare'], prefix="Fare"),

               pd.get_dummies(td['Embarked'], prefix="Embarked")],

                axis = 1)

td.drop(['Sex', 'Age', 'Fare', 'Embarked'], axis=1, inplace=True)
#формируем данные для обучения и предсказания

df_train = td[~td['Survived'].isnull()]

df_test = td[td['Survived'].isnull()]

df_test.drop(['Survived'], axis=1, inplace=True)



X = df_train.drop(('Survived'), axis=1)

y = df_train['Survived'].astype(int)

df_test.shape
df_train.head()
df_train.shape[0]
X.info()
#разделяем выборку на обучающую и тестовую

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
#импортируем библиотеку и применяем метод Gradient Boosting  к обучающей выборке

from sklearn import ensemble

gbt=ensemble.GradientBoostingClassifier(n_estimators =8, max_depth=5, random_state =33)

gbt.fit(X_train, y_train)

gbt.score(X_train, y_train)
#с помощью GridSearchCV  оцениванием и подбираем лучшие параметры

from sklearn.model_selection import GridSearchCV

param_grid=[{'n_estimators':[10,15,20], 'max_depth': [10,18,20]},]

gbt1 = ensemble.GradientBoostingClassifier(random_state =33)

grid_search= GridSearchCV(gbt1, param_grid, cv=5)

grid_search.fit(X_train, y_train)

grid_search.score(X_train, y_train)
#обучаем модель с помощью XGBClassifier

from xgboost import XGBClassifier

cl=XGBClassifier(random_state =33)

n_estimators = 10

max_depth = 10

cl.fit(X_train, y_train)

cl.score(X_train, y_train)
submission = df_test
td.columns
td=td.rename(columns={'Survived ': 'Survived'})
#делаем предсказания по тестовой выборке

submission = df_test

filename='Titanic Prediction.csv'

#submission['PassengerId'] = df_test['PassengerId'] 

print(submission.shape)

submission['Survived'] = grid_search.predict(df_test)

submission[['PassengerId', 'Survived']].to_csv(filename, index=False)

print('Saved file: '+ filename)
submission.head()
submission[['PassengerId', 'Survived']].shape