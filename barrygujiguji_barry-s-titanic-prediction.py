#Nianyi Wang(Barry) The first Hw: Titanic Survival Prediction 
import pandas as pd                

import numpy as np         
import os

os.getcwd()  #I cannot input the dataset, just check the path..
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

train.head(2)

#train.info()

#test.head(2)

train.isna().sum()
test.isna().sum()
#train.describe()
train['Age'].describe()
train['Sex'].unique()
train= train.drop('Cabin',1)

train = train.drop('Embarked',1)

train = train.drop(columns = ['Name'])

train = train.drop(columns = ['Ticket'])

train = train.drop(columns = ['Age'])
train.head(2)
test= test.drop('Cabin',1)

test = test.drop('Embarked',1)

test= test.drop(columns = ['Name'])

test = test.drop(columns = ['Ticket'])

test = test.drop(columns = ['Age'])
test.head(2)
def qqq(gender):

    if gender=="male":

        gender=1

    else:

        gender=0
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)

test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)
train.head(6)
test.head(6)
X = train.drop('Survived',1)

y= train['Survived']
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

testframe = std_scaler.fit_transform(test)

testframe.shape


from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=1000)

print(X_train)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score,recall_score,confusion_matrix

from sklearn.metrics import accuracy_score

#logistic regression

logreg = LogisticRegression(solver='liblinear', penalty='l1')

logreg.fit(X_train,y_train)

predict=logreg.predict(X_test)

print(accuracy_score(y_test,predict))

print(confusion_matrix(y_test,predict))

print(precision_score(y_test,predict))

print(recall_score(y_test,predict))
#decision tree

import sklearn.tree as sk_tree

model = sk_tree.DecisionTreeClassifier(criterion='entropy',max_depth=None,min_samples_split=2,min_samples_leaf=1,max_features=None,max_leaf_nodes=None,min_impurity_decrease=0)

model.fit(X_train,y_train)

acc=model.score(X_test,y_test) 

print('accuracy:',acc)
#nueral network

import sklearn.neural_network as sk_nn

model = sk_nn.MLPClassifier(activation='tanh',solver='adam',alpha=0.0001,learning_rate='adaptive',learning_rate_init=0.001,max_iter=200)

model.fit(X_train,y_train)

acc=model.score(X_test,y_test) 

print('accuracy:',acc)