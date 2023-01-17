import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
path='../input/drug200.csv'

df=pd.read_csv(path)

df.sample(10)
len(df)
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

X[0:5]
from sklearn import preprocessing

le_sex = preprocessing.LabelEncoder()

le_sex.fit(['F','M'])

X[:,1] = le_sex.transform(X[:,1]) 





le_BP = preprocessing.LabelEncoder()

le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])

X[:,2] = le_BP.transform(X[:,2])





le_Chol = preprocessing.LabelEncoder()

le_Chol.fit([ 'NORMAL', 'HIGH'])

X[:,3] = le_Chol.transform(X[:,3]) 



X[0:5]
y = df['Drug'].values

y[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("X_train shape: {}".format(X_train.shape))

print("y_train shape: {}".format(y_train.shape))

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)

drugTree
drugTree.fit(X_train,y_train)
predTree = drugTree.predict(X_test)
print (predTree [0:8])

print (y_test [0:8])
from sklearn import metrics

import matplotlib.pyplot as plt

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))
