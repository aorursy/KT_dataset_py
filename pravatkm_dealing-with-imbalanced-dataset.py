import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score

from sklearn.utils import resample ## Used for sampling the data
cc = pd.read_csv("../input/creditcardfraud/creditcard.csv")

cc.head()
cc.shape
cc.info()
cc.Class.value_counts()
Y = cc['Class']
Y.count()
X = cc.drop(['Class'], axis = 1)

X.head()
Y.value_counts()
## Preparing the Training and test datasets

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.20, random_state = 30)
X_train.shape
Y_train.shape
X_test.shape
Y_train.shape
## Logistic Regression

lr_model = LogisticRegression(solver='liblinear').fit(X_train,Y_train)
lr_pred = lr_model.predict(X_test)
print("Logistic Regression Metrics:")

print("")

print("Accuracy Score:",accuracy_score(Y_test, lr_pred))

print("F1 Score:", f1_score(Y_test,lr_pred))

print("Recall Score:",recall_score(Y_test, lr_pred))
## Random Forest Classifier 



rf = RandomForestClassifier(n_estimators=10)
rf_model = rf.fit(X_train, Y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest Metrics:")

print("")

print("Accuracy Score:",accuracy_score(Y_test, rf_pred))

print("Recall Score:", recall_score(Y_test, rf_pred))

print("F1 Score:", f1_score(Y_test, rf_pred))
# concatenate our training data back together



X = pd.concat([X_train, Y_train], axis=1)

X.head()
not_fraud = X[X.Class==0]

fraud = X[X.Class==1]



# upsample minority

fraud_upsampled = resample(fraud,

                          replace=True, # sample with replacement

                          n_samples=len(not_fraud), # match number in majority class

                          random_state=27) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([not_fraud, fraud_upsampled])



# check new class counts

upsampled.Class.value_counts()
y_train = upsampled.Class

X_train = upsampled.drop('Class', axis = 1)
X_train.shape
y_train.shape
## Logistic Regression

lr_model2 = LogisticRegression(solver='liblinear').fit(X_train,y_train)
lr_pred2 = lr_model2.predict(X_test)
print("Logistic Regression Metrics after Oversampling minority class:")

print("")

print("Accuracy Score:",accuracy_score(Y_test, lr_pred2))

print("F1 Score:", f1_score(Y_test,lr_pred2))

print("Recall Score:",recall_score(Y_test, lr_pred2))
## Random Forest Classifier 



rf = RandomForestClassifier(n_estimators=10)
rf_model2 = rf.fit(X_train, y_train)
rf_pred2 = rf_model2.predict(X_test)
print("Random Forest Metrics after Oversampling minority class:")

print("")

print("Accuracy Score:",accuracy_score(Y_test, rf_pred2))

print("Recall Score:", recall_score(Y_test, rf_pred2))

print("F1 Score", f1_score(Y_test, rf_pred2))
from imblearn.over_sampling import SMOTE
# Separate input features and target

y = cc.Class

X = cc.drop('Class', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)



sm = SMOTE(random_state=27, ratio=1.0)

X_train, y_train = sm.fit_sample(X_train, y_train)
lr_pred_smote = LogisticRegression(solver='liblinear').fit(X_train, y_train)



smote_pred = lr_pred_smote.predict(X_test)

print("Logistic Regression Metrics after SMOTE:")

print("")

print("Accuracy Score:",accuracy_score(y_test, smote_pred))

print("F1 Score:", f1_score(y_test,smote_pred))

print("Recall Score:",recall_score(y_test, smote_pred))
rf_pred_smote = RandomForestClassifier(n_estimators=10).fit(X_train, y_train)



rf_smote_pred = rf_pred_smote.predict(X_test)

print("Random Forest Metrics after SMOTE:")

print("")

print("Accuracy Score:",accuracy_score(y_test, rf_smote_pred))

print("Recall Score:", recall_score(y_test, rf_smote_pred))

print("F1 Score", f1_score(y_test, rf_smote_pred))