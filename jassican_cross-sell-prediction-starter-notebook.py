# import Libraries

import pandas as pd

import numpy as np

import matplotlib as pyplot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# to see all the comands result in a single kernal 

%load_ext autoreload

%autoreload 2

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
# to increase no. of rows and column visibility in outputs

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)
#Upload data

train = pd.read_csv(r'../input/janatahack-crosssell-prediction/train.csv')

test = pd.read_csv(r'../input/janatahack-crosssell-prediction/test.csv')

sample_submmission = pd.read_csv(r'../input/janatahack-crosssell-prediction/sample_submission.csv')

train.shape

test.shape

sample_submmission.shape
train.head()
train.isna().sum().sum()

test.isna().sum().sum()
train['Response'].value_counts()/len(train)

train['Gender'].value_counts()

train['Vehicle_Age'].value_counts()

train['Vehicle_Damage'].value_counts()
#converting onject to int type

train['Vehicle_Age']=train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

train['Gender']=train['Gender'].replace({'Male':1,'Female':0})

train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})

test['Vehicle_Age']=test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

test['Gender']=test['Gender'].replace({'Male':1,'Female':0})

test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})
train.columns
col_1=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']
# categorical column 

cat_col=['Gender','Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
train.info()
# changing data type because cat_feature in catboost cannot be float

train['Region_Code']=train['Region_Code'].astype(int)

test['Region_Code']=test['Region_Code'].astype(int)

train['Policy_Sales_Channel']=train['Policy_Sales_Channel'].astype(int)

test['Policy_Sales_Channel']=test['Policy_Sales_Channel'].astype(int)
X=train[col_1]

y=train['Response']
len(test)/(len(test)+len(train))
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

X_t, X_tt, y_t, y_tt = train_test_split(X, y, test_size=.25, random_state=150303,stratify=y,shuffle=True)
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

catb = CatBoostClassifier()

catb= catb.fit(X_t, y_t,cat_features=cat_col,eval_set=(X_tt, y_tt),plot=True,early_stopping_rounds=30,verbose=100)

y_cat = catb.predict(X_tt)

probs_cat_train = catb.predict_proba(X_t)[:, 1]

probs_cat_test = catb.predict_proba(X_tt)[:, 1]

roc_auc_score(y_t, probs_cat_train)

roc_auc_score(y_tt, probs_cat_test)
# from lightgbm import LGBMClassifier

# lgbcl = LGBMClassifier(n_estimators=52)

# lgbcl= lgbcl.fit(X_t, y_t,eval_metric='auc',eval_set=(X_tt , y_tt),verbose=2,categorical_feature=cat_col)

# y_lgb = lgbcl.predict(X_tt)

# probs_tr = lgbcl.predict_proba(X_t)[:, 1]

# probs_te = lgbcl.predict_proba(X_tt)[:, 1]

# roc_auc_score(y_t, probs_tr)

# roc_auc_score(y_tt, probs_te)
# 85.66 on Public leaderboard

# 85.84  on pulic leaderboard C
feat_importances = pd.Series(catb.feature_importances_, index=X_t.columns)

feat_importances.nlargest(15).plot(kind='barh')

#feat_importances.nsmallest(20).plot(kind='barh')

plt.show()
cat_pred= catb.predict_proba(test[col_1])[:, 1]

sample_submmission['Response']=cat_pred
sample_submmission.head()
sample_submmission.to_csv("cat.csv", index = False)