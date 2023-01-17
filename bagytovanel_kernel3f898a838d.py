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
import csv
gender = pd.read_csv('../input/gender_submission.csv')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
gender.head()
train.head()#здесь мы видим что нашей целью является выживший столбец, который нам нужно предсказать
test.head()
train.describe()#чтобы сделать этот набор ясным,я смотрю на их описаие и типы данных для дальнейшего возможного использования
test.describe()
train.info()
test.info()
test.shape#этот набор данных имеет 11 обьектов и 418 строк
#перед моделированием мне нужно подготовить набор данных, такой как очистка от ненужных данных и заполнение недостающих данных
if test.isnull().values.any():
    print('YES')
if train.isnull().values.any():
    print('YES this one too')
#у обоих датасетов значение ноль
#Давайте попробуем это исправить
test.isnull().sum()
train.isnull().sum()
#we have missing values in Cabin and Age and Embarked
# заполнить пробелы в наборе данных
train_random_ages = np.random.randint(train["Age"].mean() - train["Age"].std(),
                                          train["Age"].mean() + train["Age"].std(),
                                          size = train["Age"].isnull().sum())#получить случайные возрасты от среднего возраста и их откланения

test_random_ages = np.random.randint(test["Age"].mean() - test["Age"].std(),
                                          test["Age"].mean() + test["Age"].std(),
                                          size = test["Age"].isnull().sum())
train["Age"][np.isnan(train["Age"])] = train_random_ages#на все пустые строки в столбце возраста
test["Age"][np.isnan(test["Age"])] = test_random_ages
train['Age'] = train['Age'].astype(int)# обновить датасет
test['Age'] = test['Age'].astype(int)
test["Fare"].fillna(test["Fare"].median(), inplace=True)#fare
train["Embarked"].fillna('S', inplace=True)
test["Embarked"].fillna('S', inplace=True)
#it is better to work with numbers so 'S', 'Q', 'C' to 0 2 1
train['Embarked_changed'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Embarked_changed'] = test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
del train['Embarked']
del test['Embarked']
hascabin = []
for i in range(len(train['Cabin'])):
    if type(train['Cabin'][i]) is not float:
        hascabin.append(0)
    else:
        hascabin.append(1)
train['AnyCabin']=hascabin
hascabin = []
for i in range(len(test['Cabin'])):
    if type(test['Cabin'][i]) is not float:
        hascabin.append(0)
    else:
        hascabin.append(1)
test['AnyCabin']=hascabin#1 hascabin 0 has no cabin for every passenger
male_female = []
for i in range(len(train['Sex'])):
    if train['Sex'][i]=='male':
        male_female.append(1)
    else:
        male_female.append(0)
train['SEX'] = male_female
male_female = []
for i in range(len(test['Sex'])):
    if test['Sex'][i]=='male':
        male_female.append(1)
    else:
        male_female.append(0)
test['SEX'] = male_female# change sex to SEX and male as 1 and female as 0
test.head()
#delete all anneccesary columns
del test['Name']
del train['Name']
del test['Sex']
del train['Sex']
del test['Ticket']
del train['Ticket']
del test['Cabin']
del train['Cabin']
del train['PassengerId']
X_train = train.drop("Survived",axis=1)
Y_train = train["Survived"]
X_test  = test.drop("PassengerId",axis=1).copy()
print(X_train.shape)
print(Y_train.shape)
X_test.shape
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
#now constract a model
#logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

result_train = logreg.score(X_train, Y_train)
result_train
#RandomForest
random_forest = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=10,
                             min_samples_leaf=1,
                             max_features='auto',
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

random_forest =RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=5, min_samples_split=2,
                           min_samples_leaf=1, max_features='auto',    bootstrap=False, oob_score=False, 
                           n_jobs=1, random_state=seed,verbose=0)

random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

result_train = random_forest.score(X_train, Y_train)
result_train
#Support vector machine
svc = SVC(C = 0.1, gamma=0.1)
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

result_train = svc.score(X_train, Y_train)
result_train
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('titanic.csv', index=False)
print('Exported to csv file')