import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.preprocessing import StandardScaler
dataframe=pd.read_csv('../input/BankNote_Authentication.csv')

dataset=dataframe.values

dataframe.head()
sns.countplot(x='class',data=dataframe)
print(dataframe.info())

dataframe.corr(method='spearman').style.background_gradient(cmap='coolwarm')
sns.pairplot(dataframe, hue="class")
X=dataframe.iloc[:,0:4].values

Y=dataframe.iloc[:,4].values

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
scaler = StandardScaler()

X = scaler.fit_transform(X)

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression



classifier=LogisticRegression(solver='liblinear',random_state=42)

classifier.fit(X_train,Y_train)



accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)

print("Accuracies:\n",accuracies)



Y_test_pred=classifier.predict(X_test)



cm=confusion_matrix(Y_test,Y_test_pred)

acc=accuracy_score(Y_test,Y_test_pred)

print("Mean Accuracy: ",accuracies.mean())
from sklearn.svm import SVC



classifier=SVC(kernel='linear')

classifier.fit(X_train,Y_train)



accuracies=cross_val_score(estimator=classifier,X=X_train,y=Y_train,cv=10)

print("Accuracies:\n",accuracies)



Y_test_pred=classifier.predict(X_test)



cm=confusion_matrix(Y_test,Y_test_pred)

acc=accuracy_score(Y_test,Y_test_pred)

print("Mean Accuracy: ",accuracies.mean())
from sklearn.svm import SVC



classifier=SVC(kernel='rbf',gamma='auto')

classifier.fit(X_train,Y_train)



accuracies=cross_val_score(estimator=classifier,X=X_test,y=Y_test,cv=10)

print("Accuracies:\n",accuracies)

print("Mean Accuracy: ",accuracies.mean())
from sklearn.ensemble import RandomForestClassifier



classifier=RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)

classifier.fit(X_train,Y_train)

accuracies=cross_val_score(estimator=classifier,X=X_test,y=Y_test,cv=10)

 

print("Accuracies:\n",accuracies)

print("Mean Accuracy: ",accuracies.mean())
from sklearn.neural_network import MLPClassifier

classifier=MLPClassifier(hidden_layer_sizes=(8,4), max_iter=8000, alpha=0.0001, solver='sgd', verbose=10,  random_state=21,tol=0.000000001)

classifier.fit(X_train,Y_train)

accuracies=cross_val_score(estimator=classifier,X=X_test,y=Y_test,cv=10)
print("Accuracies:\n",accuracies)

print("Mean Accuracy: ",accuracies.mean())