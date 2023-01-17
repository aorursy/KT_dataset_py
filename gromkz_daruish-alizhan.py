# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from scipy.stats import skew

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import matplotlib

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import seaborn as sns

import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from scipy import stats

from sklearn.metrics import confusion_matrix

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/santander-customer-transaction-prediction/sample_submission.csv', delimiter=',')

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")

train.head()
data.dropna(inplace=True)

train.dropna(inplace=True)

test.dropna(inplace=True)
data.isna().sum()

train.isna().sum()

test.isna().sum()

data.nunique()

train.nunique()

test.nunique()
# Visulazing the distibution of the data for every feature

train.hist(edgecolor='black', linewidth=1.2, figsize=(80, 80));
sns.countplot(x='target', data=train)
sns.distplot(train.var_0) 

sns.distplot(train.var_10) 

sns.distplot(train.var_20) 

sns.distplot(train.var_30) 
plt.figure(figsize=(16,6))

plt.title("Distribution of std values per rows in the train and test set")

sns.distplot(train.std(axis=1),color="blue",label='train')

sns.distplot(test.std(axis=1),color="green",label='test')

plt.legend(); plt.show()
print(train.shape, test.shape)
X, y = train.iloc[:,2:], train.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
model_lr=LogisticRegression()

model_lr.fit(X_train,y_train)
y_pred = model_lr.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, model_lr.predict(X_test))

plt.plot(logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
fpr, tpr, thresholds = roc_curve(y_test, model_lr.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')
from xgboost import XGBClassifier

import xgboost as xgb

XGB_model = xgb.XGBClassifier()



XGB_model = XGB_model.fit(X_train, y_train)



predicted= XGB_model.predict(X_test)



from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

print("XGBoost Accuracy :", accuracy_score(y_test, predicted))
roc_auc_score(y_test, predicted)
proba = XGB_model.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], linestyle='-')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
from sklearn.naive_bayes import GaussianNB
gaus = GaussianNB()

gaus.fit(X_train,y_train)
X, y = train.iloc[:,2:], train.iloc[:,1]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 123, stratify = y)
clf = tree.DecisionTreeClassifier(max_depth=10)

clf = clf.fit(X, y)
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

def print_score(clf, X_train, y_train, X_test, y_test, train=True):

    if train:

        pred = clf.predict(X_train)

        print("Train Result:\n===========================================")

        print(f"accuracy score: {accuracy_score(y_train, pred):.4f}\n")

        print(f"Classification Report: \n \tPrecision: {precision_score(y_train, pred)}\n\tRecall Score: {recall_score(y_train, pred)}\n\tF1 score: {f1_score(y_train, pred)}\n")

        print(f"Confusion Matrix: \n {confusion_matrix(y_train, clf.predict(X_train))}\n")

        

    elif train==False:

        pred = clf.predict(X_test)

        print("Test Result:\n===========================================")        

        print(f"accuracy score: {accuracy_score(y_test, pred)}\n")

        print(f"Classification Report: \n \tPrecision: {precision_score(y_test, pred)}\n\tRecall Score: {recall_score(y_test, pred)}\n\tF1 score: {f1_score(y_test, pred)}\n")

        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
print_score(clf, X_train, y_train, X_test, y_test, train=True)

print_score(clf, X_train, y_train, X_test, y_test, train=False)