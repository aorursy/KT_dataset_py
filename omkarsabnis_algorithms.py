import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/Table.csv')
print("Number of Rows = ", len(data))
data.head()
from sklearn.model_selection import train_test_split

Y = data['Output']
X = data.drop(['Output'], axis = 1)
X = pd.get_dummies(X)
trainingset_x = X[0:90000]
trainingset_y = Y[0:90000]
testingset_x = X[90000:100000]
testingset_y = Y[90000:100000]
x_train, x_val, y_train, y_val = train_test_split(trainingset_x, trainingset_y, test_size = 0.2, random_state = 0)
print("Size of Training Data:")
print("X = ", len(x_train), "Y = ", len(y_train))
print("Size of Testing Data:")
print("X = ", len(x_val), "Y = ", len(y_val))
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
x_train = x_train[0:10000]
y_train = y_train[0:10000]
x_val = x_val[0:200]
y_val = y_val[0:200]
print("Size of Training Data:")
print("X = ", len(x_train), "Y = ", len(y_train))
print("Size of Testing Data:")
print("X = ", len(x_val), "Y = ", len(y_val))
# Support Vector Machine
svm = SVC()
svm.fit(x_train, y_train)
y_pred_svm = svm.predict(x_val)
# Accuracy and Confusion Matrix
print(accuracy_score(y_val, y_pred_svm))
print(confusion_matrix(y_val, y_pred_svm))
# Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_val)
# Accuracy and Confusion Matrix
print(accuracy_score(y_val, y_pred_dt))
print(confusion_matrix(y_val, y_pred_dt))
# Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
y_pred_mnb = mnb.predict(x_val)
# Accuracy and Confusion Matrix
print(accuracy_score(y_val, y_pred_mnb))
print(confusion_matrix(y_val, y_pred_mnb))
# Logistic Regression
lr = LogisticRegression()
lr.fit(x_train, y_train)
y_pred_lr = lr.predict(x_val)
# Accuracy and Confusion Matrix
print(accuracy_score(y_val, y_pred_lr))
print(confusion_matrix(y_val, y_pred_lr))
# Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_val)
# Accuracy and Confusion Matrix
print(accuracy_score(y_val, y_pred_rfc))
print(confusion_matrix(y_val, y_pred_rfc))
xgb = XGBClassifier()
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_val)
x_train.to_csv('x_train.csv', header=False, index=False)
y_train.to_csv('y_train.csv', header=False, index=False)

