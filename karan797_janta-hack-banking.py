# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold,KFold,GridSearchCV,GroupKFold,train_test_split
# from rfpimp import *
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/train.csv')
df_test = pd.read_csv('/kaggle/input/test.csv')
df_sub = pd.read_csv('/kaggle/input/sample_submission.csv')

df_train.isnull().sum()
df_train['Loan_Amount_Requested'] = df_train['Loan_Amount_Requested'].apply(lambda x: eval(''.join(x.split(','))))
df_test['Loan_Amount_Requested'] = df_test['Loan_Amount_Requested'].apply(lambda x: eval(''.join(x.split(','))))
df_train['Closed'] = df_train['Total_Accounts'] - df_train['Number_Open_Accounts']
df_test['Closed'] = df_test['Total_Accounts'] - df_test['Number_Open_Accounts']
df_train.replace({'Length_Employed': {'1 year': '0-2 years',
                                     '2 years':'0-2 years',
                                     '3 years':'3-5 years',
                                     '4 years': '3-5 years',
                                      '5 years': '3-5 years',
                                     '6 years': '6-7 years',
                                     '7 years':'6-7 years',
                                     '8 years':'8-9 years',
                                     '9 years':'8-9 years',
                                     '< 1 year':'0-2 years'}},inplace=True)
df_test.replace({'Length_Employed': {'1 year': '0-2 years',
                                     '2 years':'0-2 years',
                                     '3 years':'3-5 years',
                                     '4 years': '3-5 years',
                                      '5 years': '3-5 years',
                                     '6 years': '6-7 years',
                                     '7 years':'6-7 years',
                                     '8 years':'8-9 years',
                                     '9 years':'8-9 years',
                                     '< 1 year':'0-2 years'}},inplace=True)
length = {'10+ years': 10,
 '2 years': 2,
 '3 years': 3,
 '< 1 year': 0,
 '5 years': 5,
 '1 year': 1,
 '4 years': 4,
 '7 years': 7,
 '6 years': 6,
 '8 years': 8,
 '9 years': 9}
len_new =  {
    '3-5 years':1,
    '6-7 years':2,
    '8-9 years':3,
    '0-2 years':0,
    '10+ years':4
}
home = {'Mortgage': 0,
 'Rent': 1,
 'Own': 2,
 'Other': 3,
 'None': 4}
income = {'VERIFIED - income': 0,
 'VERIFIED - income source': 1,
 'not verified':2}
purp = {'debt_consolidation': 0,
 'credit_card': 1,
 'home_improvement':2,
 'other': 3,
 'major_purchase': 4,
 'small_business': 5,
 'car': 6,
 'medical': 7,
 'moving': 8,
 'vacation': 9,
 'wedding': 10,
 'house': 11,
 'renewable_energy': 12,
 'educational': 13}
gender = {'Male': 0, 'Female': 1}
target_lgb = {
    1:0,
    2:1,
    3:2
}

lgb_target = {
    0:1,
    1:2,
    2:3
}
df_train['Length_Employed'] = df_train['Length_Employed'].map(len_new)
df_train['Home_Owner'] = df_train['Home_Owner'].map(home)
df_train['Home_Owner'] = df_train['Home_Owner'].fillna(-99999)
df_train['Income_Verified'] = df_train['Income_Verified'].map(income)
df_train['Purpose_Of_Loan'] = df_train['Purpose_Of_Loan'].map(purp)
df_train['Gender'] = df_train['Gender'].map(gender)
df_train['Interest_Rate'] = df_train['Interest_Rate'].map(target_lgb)
df_test['Length_Employed'] = df_test['Length_Employed'].map(len_new)
df_test['Home_Owner'] = df_test['Home_Owner'].map(home)
df_test['Home_Owner'] = df_test['Home_Owner'].fillna(-99999)
df_test['Income_Verified'] = df_test['Income_Verified'].map(income)
df_test['Purpose_Of_Loan'] = df_test['Purpose_Of_Loan'].map(purp)
df_test['Gender'] = df_test['Gender'].map(gender)
# df_test['Interest_Rate'] = df_test['Interest_Rate'].map(target_lgb)
df_train.drop('Loan_ID',axis=1,inplace=True)
df_test.drop('Loan_ID',axis=1,inplace=True)
df_train = pd.get_dummies(data=df_train,columns=['Home_Owner','Purpose_Of_Loan','Income_Verified'])
df_test = pd.get_dummies(data=df_test,columns=['Home_Owner','Purpose_Of_Loan','Income_Verified'])
df_train['Loan_Amount_Requested_New'] = df_train['Loan_Amount_Requested']
df_test['Loan_Amount_Requested_New']= df_test['Loan_Amount_Requested']
cumulative_sum = df_train.groupby('Loan_Amount_Requested_New')["Interest_Rate"].cumsum() - df_train["Interest_Rate"]
cumulative_count = df_train.groupby('Loan_Amount_Requested_New').cumcount()
df_train['Loan_Amount_Requested_New' + "_mean_target"] = cumulative_sum/cumulative_count
vals = df_train.groupby('Loan_Amount_Requested_New').agg({'Interest_Rate':['mean']})
vals.columns = [x[0] for x in vals.columns]
vals.rename(columns={'Interest_Rate':'Loan_Amount_Requested_New_mean_target'},inplace=True)
df_test = pd.merge(df_test,vals,on='Loan_Amount_Requested_New',how='left')
df_train.drop(['Loan_Amount_Requested_New'],axis=1,inplace=True)
df_test.drop(['Loan_Amount_Requested_New'],axis=1,inplace=True)
X_train = df_train.drop('Interest_Rate',axis=1)
y_train = df_train['Interest_Rate']
k =['Loan_Amount_Requested',
 'Inquiries_Last_6Mo',
 'Months_Since_Deliquency',
 'Purpose_Of_Loan_1',
 'Annual_Income',
 'Debt_To_Income',
 'Income_Verified_2',
 'Closed',
 'Income_Verified_0',
 'Purpose_Of_Loan_3',
 'Purpose_Of_Loan_0',
 'Home_Owner_1.0',
 'Length_Employed',
 'Purpose_Of_Loan_5',
 'Total_Accounts',
 'Purpose_Of_Loan_8',
 'Purpose_Of_Loan_6',
 'Number_Open_Accounts',
 'Home_Owner_0.0',
 'Purpose_Of_Loan_4',
 'Purpose_Of_Loan_7',
 'Purpose_Of_Loan_9',
 'Purpose_Of_Loan_2',
 'Purpose_Of_Loan_13',
 'Income_Verified_1',
 'Gender',
 'Home_Owner_-99999.0',
 'Purpose_Of_Loan_11',
   'Loan_Amount_Requested_New_mean_target']
splits = 5
folds = StratifiedKFold(n_splits=splits, shuffle=True, random_state=22)
# predictions = np.zeros((len(X_valid), 3))
oof_preds = np.zeros((len(df_test), 3))
feature_importance_df = pd.DataFrame()
final_preds = []
# random_state = [77,89,22,1007,1997,1890,2000,2020,8989,786,787,1999992,2021,7654]
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        print("Fold {}".format(fold_))
        X_trn,y_trn = X_train[k].iloc[trn_idx],y_train.iloc[trn_idx]
        X_val,y_val = X_train[k].iloc[val_idx],y_train.iloc[val_idx]
        clf = lgb.LGBMClassifier(random_state=22,n_jobs=-1,max_depth=-1,min_data_in_leaf=24,num_leaves=49,bagging_fraction=0.01,
                        colsample_bytree=1.0,lambda_l1=1,lambda_l2=11,learning_rate=0.1,n_estimators=5000)
        clf.fit(X_trn, y_trn, eval_metric='multi_logloss', eval_set=[(X_val,y_val)], verbose=False,early_stopping_rounds=100)
        y_val_preds = clf.predict_proba(X_val)
        final_preds.append(f1_score(y_pred=[np.argmax(x) for x in y_val_preds],y_true=y_val,average='weighted'))
#         predictions += clf.predict_proba(X_valid)
        oof_preds += clf.predict_proba(df_test[k])
#         counter = counter + 1
oof_preds = oof_preds/splits
print(sum(final_preds)/5)
df_sub['Interest_Rate'] = [np.argmax(x) for x in oof_preds]
df_sub['Interest_Rate'] = df_sub['Interest_Rate'].map(lgb_target)
df_sub.to_csv('5Final.csv',index=False)
