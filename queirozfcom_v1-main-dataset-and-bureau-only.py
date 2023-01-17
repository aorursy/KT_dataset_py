# %load /home/felipe/firstcell.py
%reload_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',1000)

import os
from tqdm import *
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,ParameterGrid

import math
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
%matplotlib inline

RAW_DATA = "../input/"
OUTPUT_DATA = ""

def encode_categorical_columns(in_df,categorical_column_names):
    
    out_df = in_df.copy()
    
    for col_name in categorical_column_names:
        column = out_df[col_name].astype(pd.api.types.CategoricalDtype())
        out_df = pd.concat([out_df,pd.get_dummies(column, prefix=col_name,dummy_na=True)],axis=1).drop([col_name],axis=1)
        
    return out_df
SAMPLE=False
# one row = one loan

if SAMPLE:
    application_train_df = pd.read_csv(RAW_DATA+"/application_train.csv").sample(frac=0.5) 
else:
    application_train_df = pd.read_csv(RAW_DATA+"/application_train.csv")
print("{:,}".format(len(application_train_df)))
application_train_df.head()
# one row = one loan
application_test_df = pd.read_csv(RAW_DATA+"/application_test.csv")
print("{:,}".format(len(application_test_df)))
application_test_df.head()
categorical_column_names_main_data = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE',  'NAME_INCOME_TYPE',  'NAME_EDUCATION_TYPE',  'NAME_FAMILY_STATUS',
    'NAME_HOUSING_TYPE',  'FLAG_MOBIL',  'FLAG_EMP_PHONE',  'FLAG_CONT_MOBILE',  'FLAG_PHONE',
    'FLAG_EMAIL',  'OCCUPATION_TYPE',  'REGION_RATING_CLIENT',  'REGION_RATING_CLIENT_W_CITY',
    'WEEKDAY_APPR_PROCESS_START',  'REG_REGION_NOT_LIVE_REGION',  'REG_REGION_NOT_WORK_REGION',
    'LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
    'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'
]
for col in categorical_column_names_main_data:
    values=np.unique(np.concatenate([application_train_df[col].fillna('nan').values,application_test_df[col].fillna('nan').values]))

    application_train_df[col]=application_train_df[col].astype(pd.api.types.CategoricalDtype(categories=values))

    application_test_df[col]=application_test_df[col].astype(pd.api.types.CategoricalDtype(categories=values))
def process_bureau():
    
    bureau_df = pd.read_csv(RAW_DATA+"/bureau.csv")
    print("{:,} ROWS".format(len(bureau_df)))
    print("{:,} COLUMNS:".format(len(bureau_df.columns)))
    print(sorted(bureau_df.columns))

    categorical_columns = ['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE']

    in_df = encode_categorical_columns(bureau_df,categorical_columns)

    aggs = {
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'CREDIT_DAY_OVERDUE': ['min','max', 'mean','sum'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean','sum'],
            'DAYS_ENDDATE_FACT': ['min','max','mean','sum'],

            'AMT_CREDIT_MAX_OVERDUE': ['min','max','mean'],
            'CNT_CREDIT_PROLONG': ['sum'],
            'AMT_CREDIT_SUM': ['min','max','mean','sum'],
            'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['min','max','mean', 'sum'],    
            'AMT_CREDIT_SUM_OVERDUE': ['mean'],

            'DAYS_CREDIT_UPDATE':['min','max','mean'],
            'AMT_ANNUITY': ['min','max', 'mean']        
        }

    for categorical_column_name in categorical_columns:
        dummy_columns = [col for col in in_df.columns if col.startswith(categorical_column_name)]

        for dummy_column in dummy_columns:
            aggs[dummy_column] = ['sum']



    in_df = in_df.groupby('SK_ID_CURR').agg(aggs).reset_index(level=0)
    
    in_df.columns = [' '.join(col).strip() for col in in_df.columns.values]
    
    in_df=in_df[['SK_ID_CURR']+[col for col in sorted(in_df.columns) if col != 'SK_ID_CURR']]
    
    return in_df
train_df = encode_categorical_columns(application_train_df,categorical_column_names_main_data)
test_df = encode_categorical_columns(application_test_df,categorical_column_names_main_data)
train_df.head(1)
bureau_df = process_bureau()
train_df = pd.merge(
    train_df,
    bureau_df,
    how='left',
    on='SK_ID_CURR'
)
test_df = pd.merge(
    test_df,
    bureau_df,
    how='left',
    on='SK_ID_CURR'
)
train_df.head(1)
XGBClassifier().get_params()
data = train_df.drop(['SK_ID_CURR','SK_ID_BUREAU','TARGET'],axis=1).values
target = train_df['TARGET'].values

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25)
param_grid = [
    {
        'n_estimators':[100,200,300],
        'max_depth':[1,2,3,4],
        'learning_rate':[0.1,0.2]
    }
]
len(ParameterGrid(param_grid))
# num_cols = 3
# num_rows = math.ceil(len(ParameterGrid(param_grid)) / num_cols)

# # create a single figure
# plt.clf()
# fig,axes = plt.subplots(num_rows,num_cols,sharey=True)
# fig.set_size_inches(num_cols*5,num_rows*5)

# for i,g in tqdm(enumerate(ParameterGrid(param_grid))):
#     clf = XGBClassifier()
#     clf.set_params(**g)
#     clf.fit(X_train,y_train)

#     y_preds = clf.predict_proba(X_test)

#     # take the second column because the classifier outputs scores for
#     # the 0 class as well
#     preds = y_preds[:,1]

#     # fpr means false-positive-rate
#     # tpr means true-positive-rate
#     fpr, tpr, _ = metrics.roc_curve(y_test, preds)

#     auc_score = metrics.auc(fpr, tpr)

#     ax = axes[i // num_cols, i % num_cols]

#     # don't print the whole name or it won't fit
#     ax.set_title(str([r"{}:{}".format(k.split('__')[1:],v) for k,v in g.items()]),fontsize=9)
#     ax.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
#     ax.legend(loc='lower right')

#     # it's helpful to add a diagonal to indicate where chance 
#     # scores lie (i.e. just flipping a coin)
#     ax.plot([0,1],[0,1],'r--')

#     ax.set_xlim([-0.1,1.1])
#     ax.set_ylim([-0.1,1.1])
#     ax.set_ylabel('True Positive Rate')
#     ax.set_xlabel('False Positive Rate')

# plt.gcf().tight_layout()
# plt.show()

param_grid = {
        'n_estimators':300,
        'max_depth':4,
        'learning_rate':0.1
    }

clf = XGBClassifier()
clf.set_params(**param_grid)

clf.fit(data,target)
data_test = test_df.drop(['SK_ID_CURR','SK_ID_BUREAU'],axis=1).values
scores = clf.predict_proba(data_test)
out_df = test_df[['SK_ID_CURR']]
out_df['TARGET'] = scores[:,1]
out_df.to_csv("v1.csv",index=False)



