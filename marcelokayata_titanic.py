# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import re
print(os.listdir("../input"))
!cat ../input/gender_submission.csv

def tratamento(dado):
    dado = dado.fillna(0)
    dado["Age"] = dado["Age"].astype(int)
    dado["Fare"] = dado["Fare"].astype(int)
    dado.drop(['Name','Ticket','Cabin','PassengerId'],inplace=True,axis=1)
    dado = pd.get_dummies(dado)
    if 'Embarked_0' in dado.columns:
        dado.drop(['Embarked_0'],inplace=True,axis=1)
        
    return dado
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
#train_data_imputed = my_imputer.fit_transform(train_data)
test = pd.read_csv('../input/test.csv')
y = df["Survived"]
df.drop(columns=['Survived'],inplace=True,axis=1)
train_data = tratamento(df)
test_data = tratamento(test)
#dummies = pd.get_dummies(train_data)
#train_data
print(len(train_data),len(test_data),len(y))
train_data.head()
#print(len(train_data),len(test_data))
#X = train_data[titanic_features]
#X.head()
#dummies = pd.get_dummies(train_data, drop_first=True)
#X = pd.get_dummies(df, drop_first=True)
X = train_data
#X.head()
print(len(X),len(y),len(test_data))
#Model Evaluation Using a Validation Set
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model2 = LogisticRegression()#dividi o train_csv
model2.fit(X_train, y_train)#treinei o modelo
print(len(X_train),len(y_train))
# generate evaluation metrics
predicted = model2.predict(X_test)
print (metrics.accuracy_score(y_test, predicted))
print(X_train.columns,test_data.columns)
test['Survived'] = model2.predict(test_data)
test[['PassengerId','Survived']]

#result[['PassengerId','Survived']]
test[['PassengerId','Survived']].to_csv('logistic_regressiong.csv', index=False)
