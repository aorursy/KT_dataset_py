from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score

import lightgbm as lgb

import seaborn as sns

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
df_train = pd.read_csv('../input/train.csv')
df_train['target'].value_counts()
X = df_train.filter(regex='var*')
y = df_train['target']
corr = df_train.corr()
display(corr[corr!=1].abs().max().max())

display(corr.abs().min().min())
sns.heatmap(corr[corr!=1])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#lr_clf = LogisticRegression(n_jobs=-1, class_weight='balanced',verbose=2)

#lr_clf.fit(X_train, y_train)
#y_pred = lr_clf.predict(X_test)

#roc_auc_score(y_test, y_pred)
#nb_clf = GaussianNB()

#nb_clf.fit(X_train, y_train)
#y_pred = nb_clf.predict(X_test)

#roc_auc_score(y_test, y_pred)
lgb_clf = lgb.LGBMClassifier(max_depth=-1,

                             n_estimators=1000,

                             learning_rate=0.4,

                             boosting_type='dart',

                             colsample_bytree=0.3,

                             num_leaves=3,

                             metric='auc',

                             objective='binary', 

                             n_jobs=-1)
lgb_clf.fit(X_train, y_train, 

            eval_set=[(X_test, y_test)],

            verbose=0,

            early_stopping_rounds=1000)
y_pred = lgb_clf.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_pred)
df_test = pd.read_csv('../input/test.csv')

X_pred = df_test.filter(regex='var*')

predictions = lgb_clf.predict_proba(X_pred)[:,1]

pd.DataFrame({"ID_code": df_test.ID_code.values, 'target':predictions}).to_csv("submission.csv", index=False)