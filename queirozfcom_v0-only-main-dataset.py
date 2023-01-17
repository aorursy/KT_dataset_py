# %load /home/felipe/firstcell.py
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',1000)

import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
%matplotlib inline

RAW_DATA = "../input"
# one row = one loan
train_df = pd.read_csv("../input/application_train.csv")
train_df.head()
train_df['TARGET'].mean()
# one row = one loan
test_df = pd.read_csv(RAW_DATA+"/application_test.csv")
test_df.head()
categorical_column_names = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE',  'NAME_INCOME_TYPE',  'NAME_EDUCATION_TYPE',  'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',  'FLAG_MOBIL',  'FLAG_EMP_PHONE',  'FLAG_CONT_MOBILE',  'FLAG_PHONE',
    'FLAG_EMAIL',  'OCCUPATION_TYPE',  'REGION_RATING_CLIENT',  'REGION_RATING_CLIENT_W_CITY',
    'WEEKDAY_APPR_PROCESS_START',  'REG_REGION_NOT_LIVE_REGION',  'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'
]
for col in categorical_column_names:
    values=np.unique(np.concatenate([train_df[col].fillna('N/A').values,test_df[col].fillna('N/A').values]))

    train_df[col]=train_df[col].astype(pd.api.types.CategoricalDtype(categories=values))
    
    test_df[col]=test_df[col].astype(pd.api.types.CategoricalDtype(categories=values))
for col in categorical_column_names:
    train_df = pd.concat([train_df,pd.get_dummies(train_df[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
data = train_df.drop(['SK_ID_CURR','TARGET'],axis=1).values
target = train_df['TARGET'].values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
clf = XGBClassifier()
clf.fit(X_train,y_train)
y_preds = clf.predict_proba(X_test)

# take the second column because the classifier outputs scores for
# the 0 class as well
preds = y_preds[:,1]

# fpr means false-positive-rate
# tpr means true-positive-rate
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

auc_score = metrics.auc(fpr, tpr)

plt.title('ROC Curve')
plt.plot(fpr, tpr, label='AUC = {:.5f}'.format(auc_score))

# it's helpful to add a diagonal to indicate where chance 
# scores lie (i.e. just flipping a coin)
plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.legend(loc='lower right')
plt.show()
clf = XGBClassifier()
clf.fit(data,target)
for col in categorical_column_names:
    test_df = pd.concat([test_df,pd.get_dummies(test_df[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
data = test_df.drop(['SK_ID_CURR'],axis=1).values
data.shape
scores = clf.predict_proba(data)
out_df = test_df[['SK_ID_CURR']]
out_df['TARGET'] = scores[:,1]
out_df.to_csv('v0.csv', index=False)