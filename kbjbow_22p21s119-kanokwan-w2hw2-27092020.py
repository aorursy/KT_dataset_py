import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score
dataset_train = pd.read_csv('../input/titanic/train.csv')
dataset_train.head()
dataset_train.info()
dataset_train.isnull().sum()
dataset_train = dataset_train.drop(["Cabin","Embarked"], axis = 1)

dataset_train.tail()
dataset_train.isnull().sum()
dataset_train['Age'].fillna(0, inplace=True)
dataset_train.tail()
dataset_train = pd.concat([dataset_train,pd.get_dummies(dataset_train['Sex'],prefix='Sex',dummy_na=True)],axis=1).drop(['Sex'],axis=1)

dataset_train.head()
X = dataset_train[['Pclass','Age','Sex_female','Sex_male','Fare']]

Y = dataset_train[['Survived']]

X = np.array(X)

Y = np.array(Y)

print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1) 

y_test = np.array(y_test)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score



Tree_model = DecisionTreeClassifier()

Tree_model.fit(X_train,y_train)

y_pred = Tree_model.predict(X_test)



print("Accuracy:",accuracy_score(y_test, y_pred))

print("recall:",recall_score(y_test, y_pred,pos_label=1),"precision:",precision_score(y_test, y_pred,pos_label=1),"f1:",f1_score(y_test, y_pred,pos_label=1))

print("recall:",recall_score(y_test, y_pred,pos_label=0),"precision:",precision_score(y_test, y_pred,pos_label=0),"f1:",f1_score(y_test, y_pred,pos_label=0))
from sklearn.naive_bayes import GaussianNB

Gaus_model = GaussianNB()

Gaus_model.fit(X_train,y_train)

y_pred = Gaus_model.predict(X_test)



print("Accuracy:",accuracy_score(y_test, y_pred))

print("recall:",recall_score(y_test, y_pred,pos_label=1),"precision:",precision_score(y_test, y_pred,pos_label=1),"f1:",f1_score(y_test, y_pred,pos_label=1))

print("recall:",recall_score(y_test, y_pred,pos_label=0),"precision:",precision_score(y_test, y_pred,pos_label=0),"f1:",f1_score(y_test, y_pred,pos_label=0))
from sklearn.neural_network import MLPClassifier

Mlp_model = MLPClassifier(hidden_layer_sizes=(15,10,10), max_iter=300,activation = 'relu',solver='adam',random_state=1)

Mlp_model.fit(X_train,y_train)

y_pred = Mlp_model.predict(X_test)



print("Accuracy:",accuracy_score(y_test, y_pred))

print("recall:",recall_score(y_test, y_pred,pos_label=1),"precision:",precision_score(y_test, y_pred,pos_label=1),"f1:",f1_score(y_test, y_pred,pos_label=1))

print("recall:",recall_score(y_test, y_pred,pos_label=0),"precision:",precision_score(y_test, y_pred,pos_label=0),"f1:",f1_score(y_test, y_pred,pos_label=0))
def Model(model,X_train, Y_train, X_test, Y_test):

    model.fit(X_train,Y_train)

    Y_pred = model.predict(X_test)

    

    print("Accuracy:",accuracy_score(Y_test, Y_pred))

    print("recall:",recall_score(Y_test, Y_pred,pos_label=1),"precision:",precision_score(Y_test, Y_pred,pos_label=1),"f1:",f1_score(Y_test, Y_pred,pos_label=1))

    print("recall:",recall_score(Y_test, Y_pred,pos_label=0),"precision:",precision_score(Y_test, Y_pred,pos_label=0),"f1:",f1_score(Y_test, Y_pred,pos_label=0))

    return float(f1_score(Y_test, Y_pred))
def Sum(num):

    Total = num

    Total =+ Total

    return Total



def Average(num):

    N = num

    return N/5
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)



T = 0

G = 0

M = 0



count = 1

for train_index, test_index in kf.split(X):

    print("Fold: ", count)

    count += 1

    X_train, X_test = X[train_index], X[test_index]

    Y_train, Y_test = Y[train_index], Y[test_index]

    Y_train = Y_train.reshape(len(Y_train))

    

    Tree_model = DecisionTreeClassifier()

    Gaus_model = GaussianNB()

    Mlp_model = MLPClassifier(hidden_layer_sizes=(15,10,10), max_iter=300,activation = 'relu',solver='adam',random_state=1)

    

    print("Decision Tree: ")

    T = Model(Tree_model, X_train, Y_train, X_test, Y_test)

    T = Sum(T)

    print("Naive Bays")

    G = Model(Gaus_model, X_train, Y_train, X_test, Y_test)

    G = Sum(G)

    print("Neuron Network")

    M = Model(Mlp_model, X_train, Y_train, X_test, Y_test)

    M=Sum(M)

    print("\n")

    

print("Average F1 of Decision Tree model:",Average(T))

print("Average F1 of Naive Bays model:",Average(G))

print("Average F1 of Neuron Network model:",Average(M))