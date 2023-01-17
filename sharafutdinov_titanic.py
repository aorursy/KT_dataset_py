# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

%matplotlib inline
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print ('Datasets:' , 'train:' , train.shape , 'test:' , test.shape)
train.head()
train.describe()
print('TRAIN\n', train.isna().sum())
print('TEST\n', test.isna().sum())
# Смотрим попарную корреляцию
sns.heatmap(train[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# Зависимость выживания от категориальных признаков
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=train)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=train, hue='Pclass')
sns.catplot(x='SibSp', y='Survived', data=train, kind='bar')

# Обрабатываем null

# Fare (в train нет nullов)
train.dropna(subset = ['Fare'], inplace = True)
test.Fare.fillna(test.Age.median(), inplace=True)

# Age
train.Age.fillna(train.Age.median(), inplace=True)
test.Age.fillna(test.Age.median(), inplace=True)
test.shape
# Убираем ненужные признаки
train = train.drop(['PassengerId','Name','Ticket'], axis=1)
test = test.drop(['Name','Ticket'], axis=1)
# Обрабатываем категориальные признаки
[train, test] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [train, test]]
# Преобразуем признаки и добавим новые: 
# Cabin - HasCabin (1 или 0)
# FamilySize - размер семьи
# IsAlone - (1 или 0)

for df in [train, test]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)

[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [train, test]]
# Смотрим преобразованные датасеты
print(train.columns.values)
print(test.columns.values)
# Делим train на train и test
from sklearn.model_selection import train_test_split

X = train[['Age','Fare','Pclass_1', 'Pclass_2','Pclass_3', 'Sex_female',
 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin', 'FamilySize',
 'IsAlone']]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=train.Survived)
print(X_train.shape, X_test.shape)

# К соседей
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
# Обучаемся на всей выборке
classifier.fit(X, y)
# Проверяем на тесте
X_testing = test[['Age','Fare','Pclass_1', 'Pclass_2','Pclass_3', 'Sex_female',
 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'HasCabin', 'FamilySize',
 'IsAlone']]
y_test = classifier.predict(X_testing)
# Создаем файл с ответами
validation_pId = test.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_test})
my_submission.to_csv('submission_1.csv', index=False, encoding='utf8')
print(my_submission['Survived'].value_counts())
