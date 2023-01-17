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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option('display.max_columns',None)

%matplotlib inline
data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

print(data.shape)

data.head()


cat_var = [col for col in data.columns if data[col].dtype =='O']

cat_var
data.Class.value_counts()
(data.Class.value_counts()/len(data))*100
fraud =data[data['Class']==1]

not_fraud=data[data['Class']==0][:492]

df =pd.concat([fraud, not_fraud])

df.Class.value_counts()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df['Time'] = sc.fit_transform(df['Time'].values.reshape(-1,1))

df['Amount'] = sc.transform(df['Amount'].values.reshape(-1,1))
df.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('Class', axis =1), df.Class, test_size =0.2, random_state =0)

X_train.shape, X_test.shape
from sklearn.decomposition import PCA,TruncatedSVD

pca =PCA(n_components=2)

X_train_pca= pca.fit_transform(X_train)

X_test_pca = pca.transform(X_test)

pca.explained_variance_ratio_
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression



logit_model = LogisticRegression()

logit_model.fit(X_train_pca, y_train)



pred = logit_model.predict_proba(X_train_pca)



print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))



pred = logit_model.predict_proba(X_test_pca)



print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
import xgboost as xgb

xgb_model = xgb.XGBClassifier(n_estimators=100)



xgb_model = xgb.XGBClassifier()

xgb_model.fit(X_train_pca, y_train)



pred = xgb_model.predict_proba(X_train_pca)

print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))



pred = xgb_model.predict_proba(X_test_pca)

print('xgb test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
from sklearn.ensemble import AdaBoostClassifier

ada_model = AdaBoostClassifier()

ada_model.fit(X_train_pca, y_train)



pred = ada_model.predict_proba(X_train_pca)

print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = ada_model.predict_proba(X_test_pca)

print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))