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
data=pd.read_csv("/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv")

data.head()
data.isnull().sum()
data.describe()
import seaborn as sns

sns.countplot(data=data,x="diagnosis")
x=data.iloc[:,0:5]

y=data.iloc[:,5:]
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=142)
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

ypred=rfc.predict(x_test)

import sklearn.metrics as metrik

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
from sklearn.linear_model import LogisticRegression

logistic=LogisticRegression()

logistic.fit(x_train,y_train)

ypred=logistic.predict(x_test)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
from eli5.sklearn import PermutationImportance

import eli5

import shap 
perm = PermutationImportance(rfc, random_state=12).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.to_list())
perm = PermutationImportance(logistic, random_state=12).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.to_list())
from tpot import TPOTClassifier

tpot = TPOTClassifier(verbosity=2,max_time_mins=6)

tpot.fit(x_train, y_train)

print(tpot.score(x_test, y_test))
perm = PermutationImportance(tpot, random_state=12).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.to_list())
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)



ypred=xgb.predict(x_test)

print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
perm = PermutationImportance(xgb, random_state=12).fit(x_test, y_test)

eli5.show_weights(perm, feature_names = x_test.columns.to_list())