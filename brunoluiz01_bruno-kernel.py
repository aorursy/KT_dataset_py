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
df_train = pd.read_csv('../input/iris-train.csv')

df_test = pd.read_csv('../input/iris-test.csv')

df_train.head(5)
def conversion(a):

    if a== 'Iris-setosa':

        return 0

    elif a== 'Iris-versicolor':

        return 1

    elif a == 'Iris-virginica':

        return 2
def reverse(b):

    if b== 0:

        return 'Iris-setosa'

    elif b== 1:

        return 'Iris-versicolor'

    elif b == 2:

        return 'Iris-virginica' 
df_train['tgt'] = list(map(conversion,df_train['Species']))

df_train.head(5)
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import normalize

modelo = LogisticRegression()
X = df_train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

Y = df_train['tgt']
X_test = df_test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
from sklearn.model_selection import KFold
kf=KFold(n_splits=5, random_state=None, shuffle=True)

split = 1

sum_score=0

for train_index, test_index in kf.split(X,Y):

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    modelo.fit(x_train,y_train)

    score = modelo.score(x_test,y_test)

    print('Split: ' + str(split)+' - Score: '+str(score))

    split+=1

    sum_score+=score

print('Média do Score: '+str(sum_score/split))
from sklearn.neural_network import MLPClassifier



clf = MLPClassifier(activation='logistic',solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(8, 4), random_state=1)



kf=KFold(n_splits=5, random_state=None, shuffle=True)

split = 1

sum_score=0

for train_index, test_index in kf.split(X,Y):

    x_train, x_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]

    clf.fit(x_train,y_train)

    score = clf.score(x_test,y_test)

    print('Split: ' + str(split)+' - Score: '+str(score))

    split+=1

    sum_score+=score

print('Média do Score: '+str(sum_score/split))
modelo.fit(X,Y)
df_test['tgt'] = modelo.predict(X_test)
df_test['Species'] = list(map(reverse,df_test['tgt']))
modelo.score(df_test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']],df_test['tgt'])
df_final = df_test[['Id','Species']]
import sys

df_final.to_csv(sys.stdout)