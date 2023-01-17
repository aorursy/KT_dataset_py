# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/train.csv")

test_df = pd.read_csv("/kaggle/input/santander-customer-transaction-prediction/test.csv")
train_df.head()
print(train_df.shape, test_df.shape)
print(train_df.isna().sum().sum(), test_df.isna().sum().sum())
train_df.describe()
test_df.describe()
sns.set(style="darkgrid")

sns.countplot(train_df['target'])
train_df["target"].value_counts()
sns.distplot(train_df[train_df.columns[2:]].mean(), kde=False)
train_df.info()
column_corr = train_df.corr()['target']
column_corr
print(column_corr.sort_values().tail(11))
print(column_corr.sort_values().head(10))
X = train_df.iloc[:,2:202]

y = train_df.iloc[:,1]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.fit_transform(X)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=0,)

logreg.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix



y_pred = logreg.predict(X_test)

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import roc_auc_score



roc_auc_score(y_test, y_pred)
logreg.score(X_test, y_test)
y_pred = logreg.predict(test_df.drop(columns = ['ID_code']))
from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



proba = logreg.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
submission_logreg = pd.DataFrame({ "ID_code": test_df["ID_code"], "target": y_pred })

submission_logreg.to_csv('submission_logreg.csv', index=False)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
model = GaussianNB()

model.fit(X_train, y_train)

predicted= model.predict(X_test)

print("NBGaussian Accuracy :", accuracy_score(y_test, predicted))
roc_auc_score(y_test, predicted)
proba = model.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
from sklearn import tree

Tree = tree.DecisionTreeClassifier()

Tree = Tree.fit(X_train,y_train)
predicted= Tree.predict(X_test)

print("Decision Tree Accuracy :", accuracy_score(y_test, predicted))
roc_auc_score(y_test, predicted)
proba = Tree.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
from sklearn.ensemble import RandomForestClassifier

Forest = RandomForestClassifier(n_estimators = 100)

Forest = Forest.fit(X_train,y_train)
predicted= Forest.predict(X_test)

print("Random Forest Accuracy :", accuracy_score(y_test, predicted))
roc_auc_score(y_test, predicted)
proba = Forest.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()
import xgboost as xgb

XGB_model = xgb.XGBClassifier()



XGB_model = XGB_model.fit(X_train, y_train)



predicted= XGB_model.predict(X_test)



print("XGBoost Accuracy :", accuracy_score(y_test, predicted))



roc_auc_score(y_test, predicted)
proba = XGB_model.predict_proba(X_test)[:, 1]

score = roc_auc_score(y_test, proba)

fpr, tpr, _  = roc_curve(y_test, proba)



plt.figure()

plt.plot(fpr, tpr, color='c', label=f"ROC curve (auc = {score})")

plt.plot([0, 1], [0, 1], color='m', linestyle='--')

plt.title("Results")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.legend(loc="lower right")

plt.show()