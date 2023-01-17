import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score 

from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve ,auc , log_loss ,  classification_report 

from sklearn.preprocessing import StandardScaler , Binarizer

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import lightgbm as lgb

from lightgbm import LGBMClassifier

import time

import os, sys, gc, warnings, random, datetime

import math

import shap

import joblib

warnings.filterwarnings('ignore')



import xgboost as xgb

from sklearn.model_selection import StratifiedKFold , cross_val_score

from sklearn.metrics import roc_auc_score
df = pd.read_csv('../input/loan-raw/loan_test.csv',low_memory=False, index_col=0)
df.head()
df['loan_status'].value_counts()
df.head()
df = df[~df['loan_status'].isin(['Issued',

                                 'Does not meet the credit policy. Status:Fully Paid',

                                 'Does not meet the credit policy. Status:Charged Off','Current',

                                 'Late (31-120 days)', 'In Grace Period','Late (16-30 days)'

                                ])]
# 'Current',
df['loan_status'].value_counts()
def CreateDefault(Loan_Status):

    if Loan_Status in ['Default','Charged Off']:

        return 1

    else:

        return 0 

    

df['Loan_status'] = df['loan_status'].apply(lambda x: CreateDefault(x))
df['Loan_status'].value_counts()
df_default = df[df['Loan_status'] == 1]

df_FullyPaid = df[df['Loan_status'] == 0]
f,ax=plt.subplots(1,2,figsize=(18,8))

df['Loan_status'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('Loan_status')

ax[0].set_ylabel('')

sns.countplot('Loan_status',data=df,ax=ax[1])

ax[1].set_title('Loan_status')

plt.show()
df.groupby(['home_ownership','Loan_status'])['Loan_status'].count()
df.groupby(['term','Loan_status'])['Loan_status'].count()
df.isnull().sum()
df.head()
df.info()
df.isnull().sum()
df.drop('earliest_cr_line', axis=1, inplace=True) 

df.drop('mths_since_last_delinq', axis=1, inplace=True) 

df.drop('mths_since_last_record', axis=1, inplace=True) 

df.drop('last_pymnt_d', axis=1, inplace=True) 

df.drop('inq_last_6mths', axis=1, inplace=True)

df.drop(['next_pymnt_d','final_d'], axis=1, inplace=True)

df.drop('emp_length', axis=1, inplace=True)

df.drop(['year' , 'issue_d', 'loan_condition' , 'loan_condition_int' ,'loan_status'],axis=1,inplace = True)

df.drop('complete_date', axis = 1, inplace = True)

df.drop(['collections_12_mths_ex_med','last_credit_pull_d'], axis=1,inplace=True)

# fill in missing values with a specified value

df['emp_length_int'].fillna(value= 6, inplace=True)

# fill in missing values with a specified value

df['revol_util'].fillna(value= 55.06, inplace=True)

df.isnull().sum()
df.info()
df.columns.sort_values()
df = df[['acc_now_delinq', 'addr_state', 'annual_income', 'application_type',

       'collection_recovery_fee', 'delinq_2yrs', 'dti',

       'emp_length_int', 'funded_amount', 'grade', 'home_ownership',

       'income_category', 'initial_list_status', 'installment',

       'interest_payments', 'interest_rate', 'investor_funds', 

       'last_pymnt_amnt', 'loan_amount', 'open_acc', 'out_prncp',

       'out_prncp_inv', 'policy_code', 'pub_rec', 'purpose', 'pymnt_plan',

       'recoveries', 'region', 'revol_bal', 'revol_util', 'sub_grade', 'term',

       'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_rec_int',

       'total_rec_late_fee', 'total_rec_prncp', 'verification_status','Loan_status']]
# create the 'interest_payments' dummy variable using the 'map' method

df['interest_payment'] = df.interest_payments.map({'Low':1, 'High':2})

df['application_type'] = df.application_type.map({'INDIVIDUAL':1, 'JOINT':2})

df['verification_status'] = df.verification_status.map({'Verified':1, 'Source Verified':2, 'Not Verified':3})

df['home_ownership'] = df.home_ownership.map({'RENT':1, 'OWN':2, 'MORTGAGE':3, 'OTHER':4, 'NONE':5, 'ANY':6})

df['grade'] = df.grade.map({'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7})

df['term'] = df.term.map({' 36 months':1, ' 60 months':2})

df['income_category'] = df.income_category.map({'Low':1, 'Medium':2, 'High':3})

df['purpose'] = df.purpose.map({'credit_card':1, 'car':2, 

                                            'small_business':3, 'other':4,

                                            'wedding':5, 'debt_consolidation':6,

                                            'home_improvement':7, 'major_purchase':8,

                                            'medical':9, 'moving':10,

                                            'vacation':11, 'house':12,

                                            'renewable_energy':13, 'educational':14})



df['income_category'] = df['income_category'].astype(int)

df['home_ownership'] = df['home_ownership'].astype(int)

df['verification_status'] = df['verification_status'].astype(int)

df['purpose'] = df['purpose'].astype(int)

df['grade'] = df['grade'].astype(int)

df['term'] = df['term'].astype(int)

df['application_type'] = df['application_type'].astype(int)

# df['interest_payment'] = df['interest_payment'].astype(int)
df.addr_state.unique()
df.initial_list_status.unique()
df.interest_payments.unique()
df.pymnt_plan.unique()
df.region.unique()
df.sub_grade.unique()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df['addr_state'] = le.fit_transform(df['addr_state'].astype(str))

df['initial_list_status'] = le.fit_transform(df['initial_list_status'].astype(str))

df['interest_payments'] = le.fit_transform(df['interest_payments'].astype(str))

df['pymnt_plan'] = le.fit_transform(df['pymnt_plan'].astype(str))

df['region'] = le.fit_transform(df['region'].astype(str))

df['sub_grade'] = le.fit_transform(df['sub_grade'].astype(str))

df.info()
df.columns
df = df[['acc_now_delinq', 'addr_state', 'annual_income', 'application_type',

       'collection_recovery_fee', 'delinq_2yrs', 'dti',

       'emp_length_int', 'funded_amount', 'grade', 'home_ownership',

       'income_category', 'initial_list_status', 'installment',

        'interest_rate', 'investor_funds', 'loan_amount', 'open_acc', 'policy_code', 'pub_rec', 'purpose', 'pymnt_plan','recoveries',

        'region', 'revol_bal', 'revol_util', 'sub_grade', 'term',

       'total_acc',  'verification_status',

       'interest_payment' , 'Loan_status']]
X = df.drop('Loan_status', axis=1)

y = df['Loan_status']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 2020, stratify = y)
#### Export





df.to_pickle('df_pp.pkl')