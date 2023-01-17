import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import RidgeClassifier, LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC, LinearSVC , NuSVC

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()
df.groupby("SARS-Cov-2 exam result").count()
df['SARS-Cov-2 exam result'].value_counts().plot(kind='pie', autopct='%.2f%%')
sns.countplot(x='SARS-Cov-2 exam result', data=df);
Y = df['SARS-Cov-2 exam result']

Y = np.array([1 if status=="positive" else 0 for status in Y])

df = df.drop(columns=['Patient ID'])

df = df.drop(columns=['SARS-Cov-2 exam result'])
df = df.dropna(axis=1, how='all')   

df = df.dropna(axis=0, how='all')

df = df.dropna(thresh=2)

df.head()
kfold = StratifiedKFold(n_splits=30, random_state=100) # divizão equilibrada

kfold
df.dropna(axis=1, how='all')

df.head()
X = df._get_numeric_data()

X = np.nan_to_num(X.to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.333, random_state=45)
from xgboost import XGBClassifier



model = XGBClassifier()

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)

print("Classification Report")

print(metrics.classification_report(y_test, predictions,digits=4))
knn = KNeighborsClassifier(n_neighbors= 18 , weights= 'distance')

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)



print("Classification Report")

print(metrics.classification_report(y_test, y_pred,digits=4))
svc = SVC(C = 100, gamma=0.001,kernel='rbf')

svc.fit(X_train,y_train)

y_pred = svc.predict(X_test)

print("Classification Report")

print(metrics.classification_report(y_test, y_pred, digits=4))
from sklearn.tree import DecisionTreeClassifier
tree_para = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,25,30,40,50,70,90,120,150]}

clf = GridSearchCV(DecisionTreeClassifier(), tree_para, cv=kfold,scoring='accuracy')

clf.fit(X_train,y_train)
print(clf.best_params_)
tree =DecisionTreeClassifier(criterion = 'entropy',max_depth = 5 )

tree.fit(X_train,y_train)

y_pred = tree.predict(X_test)



print("Classification Report")

print(metrics.classification_report(y_test, y_pred,digits=4))
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc



gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=50, max_depth = 4, random_state = 20)

gb.fit(X_train, y_train)

predictions = gb.predict(X_test)



print("Classification Report")

print(classification_report(y_test, predictions,digits=4))