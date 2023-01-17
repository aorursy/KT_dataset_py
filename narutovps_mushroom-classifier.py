#Importing required libraries 



import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import BernoulliNB

from sklearn.preprocessing import OneHotEncoder

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/mushroom-classification/mushrooms.csv')

data.head()
for each_class in data.columns:

    print(f"{each_class} : {data[each_class].isnull().sum()}")
X = data.drop('class', axis = 1)

Y = data['class']
enc = OneHotEncoder()

X_trans = enc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_trans, Y, test_size = 0.3, random_state = 7)
KNN = KNeighborsClassifier(n_neighbors = 10)

LR = LogisticRegression(penalty='l2')

NB = BernoulliNB()



models = {'knn':KNN, 'lr': LR, 'nb':NB}



def train(model, X, Y):

    model.fit(X, Y)

        

def predict(model, X_test):

    return model.predict(X_test)



def accuracy(Y_pred, Y_test):

    return accuracy_score(Y_test, Y_pred)
pred = []

for k,v in models.items():

    train(v, X_train, y_train)

    pred.append((k, accuracy(predict(v, X_test), y_test)))
for x in pred:

    print(f"{x[0]} : {x[1]*100}")
confusion_matrix(y_test, predict(KNN, X_test)) #Confusion matrix for KNN
confusion_matrix(y_test, predict(LR, X_test)) #Confusion matrix for Logistic Regression
confusion_matrix(y_test, predict(NB, X_test)) ##Confusion matrix for Naive Bayes