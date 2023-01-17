#import libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
#read datasets

red = pd.read_csv('../input/wne-qualty-by-uci/WineQuality-RedWine.csv')

white = pd.read_csv('../input/wne-qualty-by-uci/WineQuality-WhiteWine.csv')

red['color'] = 'r'

white['color'] = 'w'

data = pd.concat([red,white],axis = 0)

col = pd.get_dummies(data['color'],drop_first=True)

data = pd.concat([data,col],axis = 1)

data = data.drop(['color'],axis=1)

data = data.sample(frac=1).reset_index(drop=True)

data.style.hide_index()

y = data['w']

x = data.drop(['w'],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
#results comparision

result = {}
#ada boost algorithm

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

ada_model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)

ada_model.fit(x_train, y_train)

ada_pred = ada_model.predict(x_test)

print("Accuracy of ada model: ",accuracy_score(y_test, ada_pred))

print("Confusion matrix:",confusion_matrix(y_test, ada_pred))

print(classification_report(y_test, ada_pred))

result['ada'] = accuracy_score(y_test, ada_pred)
#logistic regression

from sklearn.linear_model import LogisticRegression

lreg_model = LogisticRegression(random_state=0)

lreg_model.fit(x_train ,y_train)

lreg_pred = lreg_model.predict(x_test)

print("Accuracy of Logistic model: ",accuracy_score(y_test, lreg_pred))

print("Confusion matrix:",confusion_matrix(y_test, lreg_pred))

print(classification_report(y_test, lreg_pred))

result['log'] = accuracy_score(y_test, lreg_pred)
#SVM classifier

from sklearn.svm import SVC

svm_model = SVC(kernel='linear')

svm_model.fit(x_train, y_train)

svm_pred = svm_model.predict(x_test)

print("Accuracy of SVM model: ",accuracy_score(y_test, svm_pred))

print("Confusion matrix:",confusion_matrix(y_test, svm_pred))

print(classification_report(y_test, svm_pred))

result['svm'] = accuracy_score(y_test, svm_pred)
#Naive bayesian Multinomial

from sklearn.naive_bayes import MultinomialNB

mnb_model = MultinomialNB()

mnb_model.fit(x_train, y_train)

mnb_pred = mnb_model.predict(x_test)

print("Accuracy of Multinomial Naive bayesian model: ",accuracy_score(y_test, mnb_pred))

print("Confusion matrix:",confusion_matrix(y_test, mnb_pred))

print(classification_report(y_test, mnb_pred))

result['mnb'] = accuracy_score(y_test, mnb_pred)
#Gaussian naive bayesian

from sklearn.naive_bayes import GaussianNB

gnb_model = GaussianNB()

gnb_model.fit(x_train, y_train)

gnb_pred = gnb_model.predict(x_test)

print("Accuracy of Gaussian Naive bayesian model: ",accuracy_score(y_test, gnb_pred))

print("Confusion matrix:",confusion_matrix(y_test, gnb_pred))

print(classification_report(y_test, gnb_pred))

result['gnb'] = accuracy_score(y_test, gnb_pred)
#Bernoulli Naive Bayesian

from sklearn.naive_bayes import BernoulliNB

bnb_model = BernoulliNB()

bnb_model.fit(x_train, y_train)

bnb_pred = bnb_model.predict(x_test)

print("Accuracy of Bernoulli Naive bayesian model: ",accuracy_score(y_test, bnb_pred))

print("Confusion matrix:",confusion_matrix(y_test, bnb_pred))

print(classification_report(y_test, bnb_pred))

result['bnb'] = accuracy_score(y_test, bnb_pred)
#ANN 

from sklearn.neural_network import MLPClassifier

ANN_model = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)

ANN_model.fit(x_train,y_train)

ANN_pred = ANN_model.predict(x_test)

print("Accuracy of ANN model: ",accuracy_score(y_test, ANN_pred))

print("Confusion matrix:",confusion_matrix(y_test, ANN_pred))

print(classification_report(y_test, ANN_pred))

result['ann'] = accuracy_score(y_test, ANN_pred)
lists = result.items() # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples

plt.bar(x, y)

plt.ylim(top = 1.0)

plt.ylim(bottom = 0.75)

plt.show()