# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import RFE

from sklearn.decomposition import PCA

from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

import random

from sklearn import metrics

from imblearn.over_sampling import SMOTE



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/santander-customer-satisfaction/train.csv')

test = pd.read_csv('../input/santander-customer-satisfaction/test.csv')
train.head()
train.describe()
train.isna().sum().sort_values(ascending=False)
train['imp_ent_var16_ult1'].value_counts()
# Checking outliers at 25%,50%,75%,90%,95% and 99%

train.describe(percentiles=[.25,.5,.75,.90,.95, .975,.99,.999])
high = .99

first_quartile = 0.25

third_quartile = 0.75

quant_df = train.quantile([high, first_quartile, third_quartile])
quant_df
train_df = train.drop(['ID', 'TARGET'], axis = 1)
train_df = train_df.apply(lambda x: x[(x <= quant_df.loc[high,x.name])], axis=0)
train_df.describe(include='all')
train_df.shape
train_df.head()
train_df = pd.concat([train.loc[:,'ID'], train_df], axis=1)



train_df = pd.concat([train.loc[:,'TARGET'], train_df], axis=1)
train_df.describe()
train_df.isnull().sum().sort_values(ascending=False)
new_train_df = train_df

for col in new_train_df.columns:

    min_val = min(new_train_df[col])

    max_val = max(new_train_df[col])

    new_train_df[col].fillna(round(random.uniform(min_val, max_val), 2), inplace =True)
new_train_df.isna().sum().sort_values(ascending=False)
y = new_train_df['TARGET']

X = new_train_df.drop(['TARGET','ID'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
lr = LogisticRegression()

lr.fit(X_train,y_train)

preds = lr.predict(X_test)

print("Accuracy with Logistic = ", metrics.accuracy_score(y_test, preds))
new_train_df['TARGET'].value_counts()
sm = SMOTE(kind = "regular")

X_tr,y_tr = sm.fit_sample(X_train,y_train)
print(X_tr.shape)

print(y_tr.shape)

lr.fit(X_tr,y_tr)



lr_preds = lr.predict(X_test)

print("Accuracy with Logistic = ", metrics.accuracy_score(y_test, lr_preds))
dt1 = DecisionTreeClassifier(max_depth=5)

dt1.fit(X_tr, y_tr)



dt_preds = dt1.predict(X_test)

print("Accuracy with Decision Tree = ", metrics.accuracy_score(y_test, dt_preds))
rft = RandomForestClassifier(n_jobs=-1)

rft.fit(X_tr, y_tr)



rft_preds = rft.predict(X_test)

print("Accuracy with Random Forest = ", metrics.accuracy_score(y_test, rft_preds))
x_test_final = test.drop(['ID'], axis=1)

final_prediction = rft.predict(x_test_final)
submission = pd.DataFrame({

        "ID": test["ID"],

        "TARGET": final_prediction

    })

submission.to_csv('RandomForect.csv',header=True, index=False)