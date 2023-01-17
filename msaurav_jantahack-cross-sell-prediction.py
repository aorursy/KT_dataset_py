# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold

import time

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
#reading data

train=pd.read_csv("../input/janatahack-crosssell-prediction/train.csv")

test=pd.read_csv("../input/janatahack-crosssell-prediction/test.csv")

sub=pd.read_csv('../input/janatahack-crosssell-prediction/sample_submission_iA3afxn.csv')

train.shape

test.shape

sub.shape
train.head()
train.info()

test.info()
train.describe()

test.describe()
sns.countplot(train['Response']);
sns.countplot(train['Response'],hue=train['Previously_Insured']);
sns.countplot(pd.concat([train['Previously_Insured'],test['Previously_Insured']],axis=0));
sns.countplot(train['Response'],hue=train['Gender']);
sns.countplot(pd.concat([train['Gender'],test['Gender']],axis=0));
sns.countplot(train['Response'],hue=train['Driving_License']);
sns.countplot(pd.concat([train['Driving_License'],test['Driving_License']],axis=0));
len(train[train['Driving_License']==0])/len(train)
sns.countplot(train['Response'],hue=train['Vehicle_Damage']);
sns.countplot(pd.concat([train['Vehicle_Damage'],test['Vehicle_Damage']],axis=0));
sns.countplot(train['Response'],hue=train['Vehicle_Age']);
sns.countplot(pd.concat([train['Vehicle_Age'],test['Vehicle_Age']],axis=0));
train['id'].nunique()

test['id'].nunique()
sns.distplot(train['Annual_Premium']);
#Data is left Skewed as we can see from above distplot

train['Annual_Premium']=np.log1p(train['Annual_Premium'])

sns.distplot(train['Annual_Premium']);
test['Annual_Premium']=np.log1p(test['Annual_Premium'])

sns.distplot(test['Annual_Premium']);
# Missing Values

train.isnull().sum().sum()

test.isnull().sum().sum()
cat_cols=['Driving_License','Gender','Policy_Sales_Channel','Previously_Insured','Region_Code','Vehicle_Age','Vehicle_Damage']
#converting categorical features to numerical features

train['Gender']=train['Gender'].replace({'Male':1,'Female':0})

train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})

train['Vehicle_Age']=train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

test['Gender']=test['Gender'].replace({'Male':1,'Female':0})

test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})

test['Vehicle_Age']=test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})
test_id=test['id']
train=train.drop(columns=['id'],axis=1)

test=test.drop(columns=['id'],axis=1)

#cat_cols=['Gender','Policy_Sales_Channel','Previously_Insured','Region_Code','Vehicle_Age','Vehicle_Damage']
train.info()

test.info()
#Checking correlation between features

plt.figure(figsize=(10,10))

sns.heatmap(train.corr(),annot=True);
#creating X and y for training

y=train['Response']

X=train.drop(columns='Response',axis=1)
len(test)/(len(test)+len(train))
X_tr,X_te,y_tr,y_te=train_test_split(X,y,random_state=0,test_size=0.25,stratify=y,shuffle=True)
#lgbm stratified k-fold

#%%time

err = [] 

y_pred_tot_lgbm = np.zeros((len(test), 2))





fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

i = 1



for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    m = LGBMClassifier(n_estimators=61,reg_alpha=2.1,reg_lambda=2,importance_type='gain')

    m.fit(x_train, y_train,eval_set=[(x_val, y_val)], categorical_feature=cat_cols,eval_metric='auc',verbose=0)

    pred_y = m.predict_proba(x_val)[:,1]

    print(i, " err_lgm: ", roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val,pred_y))

    y_pred_tot_lgbm+= m.predict_proba(test)

    i = i + 1

y_pred_tot_lgbm=y_pred_tot_lgbm/5

sum(err)/5

#XGBoost stratified k-fold

#%%time

err = [] 

y_pred_tot_xgb = np.zeros((len(test), 2))





fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=2020)

i = 1



for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    m = XGBClassifier(n_estimators=114,reg_lambda=2,learning_rate=0.09,max_depth=7,min_child_weight=5)

    m.fit(x_train, y_train,eval_set=[(x_val, y_val)],eval_metric='auc',verbose=0)

    pred_y = m.predict_proba(x_val)[:,1]

    print(i, " err_xgb: ", roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val,pred_y))

    y_pred_tot_xgb+= m.predict_proba(test)

    i = i + 1

y_pred_tot_xgb=y_pred_tot_xgb/4

sum(err)/4
# changing data type because cat_feature in catboost cannot be float

train['Region_Code']=train['Region_Code'].astype(int)

test['Region_Code']=test['Region_Code'].astype(int)

train['Policy_Sales_Channel']=train['Policy_Sales_Channel'].astype(int)

test['Policy_Sales_Channel']=test['Policy_Sales_Channel'].astype(int)
train.info()

cat_cols

#creating X and y for training

y=train['Response']

X=train.drop(columns='Response',axis=1)

#X_tr,X_te,y_tr,y_te=train_test_split(X,y,test_size=0.25,stratify=y,shuffle=True,random_state=150303)
# CatBoost stratified k-fold

#%%time

err = [] 

y_pred_tot_catb = np.zeros((len(test), 2))





fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)

i = 1



for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    m = CatBoostClassifier(eval_metric='AUC')

    m.fit(x_train, y_train,eval_set=[(x_val, y_val)],cat_features=cat_cols

          ,plot=True,early_stopping_rounds=30,verbose=0)



    pred_y = m.predict_proba(x_val)[:,1]

    print(i, " err_catb: ", roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val,pred_y))

    y_pred_tot_catb+= m.predict_proba(test)

    i = i + 1

y_pred_tot_catb=y_pred_tot_catb/5

sum(err)/5
#weighted average of stratified xgboost and lgbm

y_avg_xgb_lgb_kfold=y_pred_tot_lgbm[:,1]*0.61+y_pred_tot_xgb[:,1]*0.39

# res=pd.concat([test_id,pd.DataFrame(y_avg_xgb_lgb_kfold,columns=['Response'])],axis=1)

# res.to_csv('avg2_kfold.csv',index=False)
#weighted average of above result and catboost

avg3_kfold=y_avg_xgb_lgb_kfold*.59+y_pred_tot_catb[:,1]*0.41

res=pd.concat([test_id,pd.DataFrame(avg3_kfold,columns=['Response'])],axis=1)

res.to_csv('avg3_kfold.csv',index=False)