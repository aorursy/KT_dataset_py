import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier

import seaborn as sns

from sklearn.metrics import roc_auc_score
train = pd.read_csv('../input/jantahack-cross-sell-prediction/train.csv')

test = pd.read_csv('../input/jantahack-cross-sell-prediction/test.csv')
train.head()
test.head()
train.info()
train["Region_Code"].nunique()
train["Vehicle_Age"].value_counts()
sns.distplot(train["Annual_Premium"],bins=40)
train["Annual_Premium"] = np.log(train["Annual_Premium"])

test["Annual_Premium"] = np.log(test["Annual_Premium"])

sns.distplot(train["Annual_Premium"])
sns.heatmap(train.drop('id',axis=1).corr(),cmap='coolwarm')
print(train["Response"].value_counts())

sns.countplot(train["Response"])
concat_df = pd.concat([train,test],axis=0)
sns.pairplot(concat_df.drop("id",axis=1))
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
#label encoding for Gender

en = LabelEncoder()

concat_df["Gender"] = en.fit_transform(concat_df["Gender"])

#label encoding for Vehicle_Damage

en = LabelEncoder()

concat_df["Vehicle_Damage"] = en.fit_transform(concat_df["Vehicle_Damage"])

# Frequency encoding for Vehicle Age

f = concat_df.groupby(concat_df["Vehicle_Age"]).size()/len(concat_df)

concat_df["Vehicle_Age"] = concat_df["Vehicle_Age"].apply(lambda x: f[x])
concat_df.head()
train_feat = concat_df[pd.isnull(concat_df["Response"])==False].drop("Response",axis=1)

train_targets = concat_df[pd.isnull(concat_df["Response"])==False]["Response"]

test_feat = concat_df[pd.isnull(concat_df["Response"])==True]
train_feat.drop("id",axis=1,inplace=True)

test_feat.drop(["id","Response"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
train_feat["Region_Code"]=train_feat["Region_Code"].astype(int)

train_feat["Policy_Sales_Channel"]=train_feat["Policy_Sales_Channel"].astype(int)

test_feat["Region_Code"]=test_feat["Region_Code"].astype(int)

test_feat["Policy_Sales_Channel"]=test_feat["Policy_Sales_Channel"].astype(int)
X_train,X_val,Y_train,Y_val = train_test_split(train_feat,train_targets,test_size=0.3,random_state=101)
lgb = LGBMClassifier(boosting_type='gbdt',n_estimators=500,max_depth=7,learning_rate=0.04,objective='binary',metric='auc',is_unbalance=True,

                 colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=101,n_jobs=-1)
lgb.fit(X_train,Y_train)
print(roc_auc_score(Y_val,lgb.predict_proba(X_val)[:,1]))
pred2 = lgb.predict_proba(test_feat)[:,1]
from xgboost import XGBClassifier
xgb = XGBClassifier(objective='binary:logistic',boosting_type = 'gbdt',n_estimators=400,max_depth=8,learning_rate=0.04,metric='auc',

                   colsample_bytree=0.5,reg_lambda=2,reg_alpha=2,random_state=101,n_jobs=-1)
xgb.fit(X_train,Y_train)
print(roc_auc_score(Y_val,xgb.predict_proba(X_val)[:,1]))
pred1 = xgb.predict_proba(test_feat)[:,1]
from catboost import CatBoostClassifier
cat = ["Gender","Driving_License","Region_Code","Previously_Insured","Vehicle_Damage","Policy_Sales_Channel"]
cbc = CatBoostClassifier(n_estimators=1000,l2_leaf_reg=2,custom_metric=['AUC','Logloss'],learning_rate = 0.04,cat_features=cat,max_depth=8,eval_metric="AUC")
cbc.fit(X_train,Y_train,eval_set=(X_val,Y_val),early_stopping_rounds=20)
pred = cbc.predict_proba(test_feat)[:,1]
n_pred = (5*pred+pred1+pred2)/7

df = pd.DataFrame({'id':test["id"],'Response':n_pred})

df.to_csv('result2.csv',index=False)