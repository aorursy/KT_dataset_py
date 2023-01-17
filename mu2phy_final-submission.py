import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV



from sklearn.svm import LinearSVC

from xgboost import XGBClassifier



import sklearn.metrics as metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import seaborn as sns



import os

print(os.listdir("../input"))
dataset = pd.read_csv(r'../input/train.csv', low_memory = True)
dataset.sample(5)
dataset.describe()
dataset.info()
dataset.target.hist()
target0 = dataset[dataset['target'] == 0].sample(15000)

target1 = dataset[dataset['target'] == 1].sample(15000)

sampleDf = pd.concat([target0,target1])
sampleDf.sample(10)
sampleDf.describe()
dataset.describe()
X = sampleDf.iloc[:,2:]

y = sampleDf.iloc[:,1]



scaler = StandardScaler()

scaler.fit_transform(X, y=None)



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
GNBlf = GaussianNB()

GNBlf.fit(X_train,y_train)

y_pred1 = GNBlf.predict(X_test)





fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred1)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
lrClf = LogisticRegression(C = 1, penalty = 'l2')

lrClf.fit(X_train, y_train)

y_pred2 = lrClf.predict(X_test)





fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred2)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
dtclf = DecisionTreeClassifier()

dtclf.fit(X_train, y_train)

y_pred3 = dtclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred3)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
"""

parameters = {

    'n_estimators': [100,200,300],

    'max_depth': [8, 9, 10, 11, 12],

    'max_features': ['auto', 'sqrt']

}



grid_search = GridSearchCV(RFclf, param_grid=parameters, cv = 2, n_jobs=-1)

grid_search.fit(X_train, y_train)

grid_search.best_params_

"""
RFclf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1)

RFclf.fit(X_train,y_train)

y_pred4 = RFclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred4)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
xgbclf = XGBClassifier(objective="binary:logistic")

xgbclf.fit(X_train, y_train)

y_pred5 = xgbclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred5)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
X2 = dataset.iloc[:,2:]

y2 = dataset.iloc[:,1]



X_train, X_test, y_train, y_test = train_test_split(X2,y2, test_size=0.25, random_state=123)



scaler = StandardScaler()

scaler.fit_transform(X2, y2)
GNBlf = GaussianNB()

GNBlf.fit(X_train,y_train)

y2_pred1 = GNBlf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred1)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
lrClf = LogisticRegression(C = 1, penalty = 'l2')

lrClf.fit(X_train, y_train)

y2_pred2 = lrClf.predict(X_test)





fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred2)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
dtclf = DecisionTreeClassifier()

dtclf.fit(X_train, y_train)

y2_pred3 = dtclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred3)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
RFclf = RandomForestClassifier(n_estimators=100, max_depth=10,n_jobs = -1)

RFclf.fit(X_train,y_train)

y2_pred4 = RFclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred4)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()
xgbclf = XGBClassifier(objective="binary:logistic")

xgbclf.fit(X_train, y_train)

y2_pred5 = xgbclf.predict(X_test)



fpr, tpr, thresholds = metrics.roc_curve(y_test, y2_pred5)

roc_auc = metrics.auc(fpr, tpr)

print(roc_auc)



plt.figure()

plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()