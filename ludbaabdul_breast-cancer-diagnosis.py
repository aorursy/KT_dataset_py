# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict
Data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

data = Data.drop('Unnamed: 32', axis = 1)

X = data.drop('diagnosis', axis = 1)

y = data.diagnosis



X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 123)

X_train.isnull().any()
X_train.head(10)
X_train.describe()
model = RandomForestClassifier(random_state = 123, n_estimators = 100)

# model = XGBClassifier(random_state = 123, n_estimators = 100, learning_rate = 0.3)

# model = SGDClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_valid)



acc = accuracy_score(y_valid, pred)

confusion_mat = confusion_matrix(y_valid, pred)

print('Accuracy: ', acc)

print('Confusion_mat:\n ', confusion_mat)



plt.matshow(confusion_mat, cmap=plt.cm.gray)

plt.xlabel('True Values')

plt.ylabel('predicted Values')

plt.title('Confusion Matrix')

plt.show()

y_train_M = (y_train == 'M') # making only type M as our training target

y_valid_M = (y_valid == 'M') # making only type M as our validation target
model.fit(X_train, y_train_M)

pred = model.predict(X_valid)



confusion_mat = confusion_matrix(y_valid_M, pred)

precision = precision_score(y_valid_M, pred)

recall = recall_score(y_valid_M, pred)

print('Accuracy: ', acc)

print('Confusion_mat:\n ', confusion_mat)

print('Precision: ', precision)

print('recall: ', recall)



plt.matshow(confusion_mat, cmap=plt.cm.gray)

plt.xlabel('True Values')

plt.ylabel('predicted Values')

plt.title('Confusion Matrix')

plt.show()
pred_scores_of_M = cross_val_predict(model, X_train, y_train_M, cv= 3, method="predict_proba")

pred_scores_of_M = pred_scores_of_M[:, 1] # score = proba of positive class

# pred_scores
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_M, pred_scores_of_M)



def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC curve')

    

plot_roc_curve(fpr, tpr)

plt.show()
y_train_B = (y_train == 'B') # making only type B as our training target

y_valid_B = (y_valid == 'B') # making only type B as our validation target
model.fit(X_train, y_train_B)

pred = model.predict(X_valid)



confusion_mat = confusion_matrix(y_valid_B, pred)

precision = precision_score(y_valid_B, pred)

recall = recall_score(y_valid_B, pred)

print('Accuracy: ', acc)

print('Confusion_mat:\n ', confusion_mat)

print('Precision: ', precision)

print('recall: ', recall)



plt.matshow(confusion_mat, cmap=plt.cm.gray)

plt.xlabel('True Values')

plt.ylabel('predicted Values')

plt.title('Confusion Matrix')

plt.show()
pred_scores_of_B = cross_val_predict(model, X_train, y_train_B, cv= 3, method="predict_proba")

pred_scores_of_B = pred_scores_of_B[:, 1] # score = proba of positive class

# pred_scores
fpr, tpr, thresholds = roc_curve(y_train_B, pred_scores_of_B)

plot_roc_curve(fpr, tpr)

plt.show()