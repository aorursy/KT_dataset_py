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
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data.shape
data.head()
data.Survived.describe() #зачем они float - не понятно, исправляем
data.Survived.describe() 
data[data.isnull().any(axis = 1)]
data.info() #тут мы видим, что почти все Cabin nan, удаляем столбец
data.drop('Cabin', axis=1, inplace=True)
corrmat = data.corr()
plt.subplots(figsize=(11, 8))
sns.heatmap(corrmat, vmax=.8, square=True);
data['Age'].fillna(data['Age'].mean(), inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data.Sex)
data.Sex = le.transform(data.Sex)
data.info()
le.fit(data_test.Sex)
data_test.Sex = le.transform(data_test.Sex)

data.head()
X = data.drop(['Survived'], 1)
y = data.Survived
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.25, random_state = 25)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
features = ['Pclass','Fare','Age','Sex']
clf1 = DecisionTreeClassifier(random_state=241)
clf1 = clf1.fit(X_train[features],y_train)
clf1.score(X_test[features],y_test) #Наше Дерево, точность существенно ниже, чем в примере с достойной подготовкой данных
clf2 = KNeighborsClassifier(n_neighbors=4)
clf2 = clf2.fit(X_train[features], y_train)
clf2.score(X_test[features],y_test) #опять существенно меньше, попробуем сделать scale
from sklearn.preprocessing import scale
X_trainSC = pd.DataFrame(scale(X_train[features]))
X_trainSC.columns = X_train[features].columns
#Сделаем новое дерево и соседей и сравним результаты
clf1SC = DecisionTreeClassifier(random_state=241)
clf1SC = clf1SC.fit(X_trainSC,y_train)
clf1SC.score(X_test[features],y_test) #только во вред( попоробуем заскейлить и тестовую
X_testSC = pd.DataFrame(scale(X_test[features])) 
X_testSC.columns = X_test[features].columns
clf1SC.score(X_testSC,y_test) #Все равно хуже, чем без scal'а 
from sklearn.linear_model import LogisticRegression
clf3 = LogisticRegression()
clf3 = clf3.fit(X_train[features],y_train)
clf3.score(X_test[features], y_test) #индусы все равно впереди, надо работать с данными
new_features = ['Pclass','Fare','Age','Sex', 'PassengerId']
X_test.head()
preds = clf3.predict(X_test[features])
submission = pd.DataFrame({
        "PassengerId": X_test.PassengerId,
        "Survived": preds
    })
submission.to_csv('titanic.csv', index=False)
