# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv('../input/titanic_data.csv')

df.shape
df.head()
df = df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)



df = df.dropna()

df.head()
sexConvDict = {"male":1 , "female" :2}

df['Sex'] = df['Sex'].apply(sexConvDict.get).astype(int)

df.head()
features = ['Sex', 'Parch', 'Pclass', 'Age', 'Fare', 'SibSp']

df[features]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X = scaler.fit_transform(df[features].values)

#X = df[features].values

y = df['Survived'].values

df.head()
X
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train.shape

df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train.shape
from sklearn.neural_network import MLPClassifier as mlp



nn = mlp(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(3, 2), random_state=0)

nn.fit(X_train, y_train)
predicted = nn.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, predicted)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, predicted)