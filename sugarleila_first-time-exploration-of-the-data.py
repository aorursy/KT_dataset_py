# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# Any results you write to the current directory are saved as output.
#application_train/test
app_train = pd.read_csv('../input/application_train.csv')
print('Testing data shape:',app_train.shape)
app_test = pd.read_csv('../input/application_test.csv')
print('Testing data shape:',app_test.shape)
app_train.head()
app_test.head()
#bureau
bureau = pd.read_csv('../input/bureau.csv')
print('Testing data shape:',bureau.shape)
bureau.head()
#bureau
bure_bal = pd.read_csv('../input/bureau_balance.csv')
print('Testing data shape:',bure_bal.shape)
bure_bal.head()
#previous_application
prev = pd.read_csv('../input/previous_application.csv')
print('Testing data shape:',prev.shape)
prev.head()
# POS_CASH_BALANCE
pos_cash = pd.read_csv('../input/POS_CASH_balance.csv')
print('Testing data shape:',pos_cash.shape)
pos_cash.head()
# credit_card_balance
cc = pd.read_csv('../input/credit_card_balance.csv')
print('Testing data shape:',cc.shape)
cc.head()
# installments_payment:
instal = pd.read_csv('../input/installments_payments.csv')
print('Testing data shape:',instal.shape)
instal.head()
#checking missing data app_train
total = app_train.isnull().sum().sort_values(ascending = False)
percent = (app_train.isnull().sum()/app_train.isnull().count()*100).sort_values(ascending = False)
missing_app_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_app_train_data.head(20)
#checking missing data app_test
total = app_test.isnull().sum().sort_values(ascending = False)
percent = (app_test.isnull().sum()/app_test.isnull().count()*100).sort_values(ascending = False)
missing_app_test_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_app_test_data.head(20)
#missing data bureau
total = bureau.isnull().sum().sort_values(ascending = False)
percent = (bureau.isnull().sum()/bureau.isnull().count()*100).sort_values(ascending = False)
missing_bureau_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bureau_data.head(20)
#missing data bureau_balance
total = bure_bal.isnull().sum().sort_values(ascending = False)
percent = (bure_bal.isnull().sum()/bure_bal.isnull().count()*100).sort_values(ascending = False)
missing_bure_bal_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_bure_bal_data.head(20)
#missing data previous_application
total = prev.isnull().sum().sort_values(ascending = False)
percent = (prev.isnull().sum()/prev.isnull().count()*100).sort_values(ascending = False)
missing_prev_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_prev_data.head(20)
#missing data POS_CASH_balance
total = pos_cash.isnull().sum().sort_values(ascending = False)
percent = (pos_cash.isnull().sum()/pos_cash.isnull().count()*100).sort_values(ascending = False)
missing_pos_cash_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_pos_cash_data.head(20)
#missing data POS_CASH_balance
total = cc.isnull().sum().sort_values(ascending = False)
percent = (cc.isnull().sum()/cc.isnull().count()*100).sort_values(ascending = False)
missing_cc_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_cc_data.head(20)
print('There are {} features in the app_train'.format(len(app_train.columns)))
print('There are {} features in the app_test'.format(len(app_test.columns)))
print('There are {} features in the bureau'.format(len(bureau.columns)))
print('There are {} features in the bure_bal'.format(len(bure_bal.columns)))
print('There are {} features in the bure_bal'.format(len(prev.columns)))
print('There are {} features in the pos_cash'.format(len(pos_cash.columns)))
print('There are {} features in the credit_card_balance'.format(len(cc.columns)))
# 3.3 Categorical features excoding
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns,dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

app_train1,new_columns = one_hot_encoder(app_train)
#Collection of target and feature
for f in new_columns:
    print('Target Correlation by:', f)
    print(app_train1[[f, 'TARGET']].groupby(f, as_index=False).sum())
    print('-'*10, '\n')
app_train['TARGET'].value_counts()
app_train1 = app_train
app_train['CODE_GENDER'].value_counts()
print(app_train['TARGET'].sum())
#Collection of target and feature
columns = [f for f in app_train.columns]
Categoric_feats = app_train.select_dtypes('object').apply(pd.Series.nunique,axis = 0)
k = app_train['TARGET'].sum()

for f in columns:
    if f in Categoric_feats:
        print('Target Correlation by:', f)
        print(app_train[[f, 'TARGET']].groupby(f, as_index=False).sum())
        print('-'*10, '\n')

#using crosstabs: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.crosstab.html
#print(pd.crosstab(app_train['Title'],app_train['TARGET']))
#Visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

#
plt.figure(figsize=[16,6])
# plt.subplot(231)
# plt.boxplot(app_train['EXT_SOURCE_1'], showmeans = True, meanline = True)
# plt.title('EXT_SOURCE_1 Boxplot')
# plt.ylabel('EXT_SOURCE_1')

# plt.subplot(232)
# plt.boxplot(app_train['EXT_SOURCE_2'], showmeans = True, meanline = True)
# plt.title('EXT_SOURCE_2 Boxplot')
# plt.ylabel('EXT_SOURCE_2 ')

# plt.subplot(233)
# plt.boxplot(app_train['EXT_SOURCE_3'], showmeans = True, meanline = True)
# plt.title('EXT_SOURCE_3 Boxplot')
# plt.ylabel('EXT_SOURCE_3 ')

plt.subplot(131)
plt.hist(x = [app_train[app_train['TARGET']==1]['EXT_SOURCE_1'], app_train[app_train['TARGET']==0]['EXT_SOURCE_1']], 
         stacked=True, color = ['g','r'],label = ['1','0'])
plt.title('EXT_SOURCE_1 have difficulties or not')
plt.xlabel('EXT_SOURCE_1 ($)')
plt.ylabel('payment difficulties')
plt.legend()

plt.subplot(132)
plt.hist(x = [app_train[app_train['TARGET']==1]['EXT_SOURCE_2'], app_train[app_train['TARGET']==0]['EXT_SOURCE_2']], 
         stacked=True, color = ['g','r'],label = ['1','0'])
plt.title('EXT_SOURCE_2 have difficulties or not')
plt.xlabel('EXT_SOURCE_2 ($)')
plt.ylabel('payment difficulties')
plt.legend()

plt.subplot(133)
plt.hist(x = [app_train[app_train['TARGET']==1]['EXT_SOURCE_3'], app_train[app_train['TARGET']==0]['EXT_SOURCE_3']], 
         stacked=True, color = ['g','r'],label = ['1','0'])
plt.title('EXT_SOURCE_3 have difficulties or not')
plt.xlabel('EXT_SOURCE_3($)')
plt.ylabel('payment difficulties')
plt.legend()
#copy data
app_train1 = app_train
def LabelPrep(df):
    x = []
    for i in df.unique():
        x.append(i)
    return x    
x_NAME_TYPE_SUITE = LabelPrep(app_train['NAME_TYPE_SUITE'])
print(x_NAME_TYPE_SUITE)
x_NAME_TYPE_SUITE1 = {}
x_NAME_TYPE_SUITE = LabelPrep(app_train['NAME_TYPE_SUITE'])
fig, saxis = plt.subplots(2, 3,figsize=(20,16))

sns.barplot(x = 'CODE_GENDER', y = 'TARGET', data=app_train, ax = saxis[0,0])
sns.barplot(x = 'NAME_TYPE_SUITE', y = 'TARGET', data=app_train, ax = saxis[0,1])
saxis[0,1].set_xticklabels(saxis[0,1].get_xticklabels(), rotation=30)
sns.barplot(x = 'NAME_HOUSING_TYPE', y = 'TARGET',data=app_train, ax = saxis[0,2])
saxis[0,2].set_xticklabels(saxis[0,2].get_xticklabels(), rotation=-30)

sns.pointplot(x = 'CODE_GENDER', y = 'TARGET', data=app_train, ax = saxis[1,0])
sns.pointplot(x = 'NAME_TYPE_SUITE', y = 'TARGET', data=app_train, ax = saxis[1,1])
saxis[1,1].set_xticklabels(saxis[1,1].get_xticklabels(), rotation=30)
sns.pointplot(x = 'NAME_HOUSING_TYPE', y = 'TARGET', data=app_train, ax = saxis[1,2])
saxis[1,2].set_xticklabels(saxis[1,2].get_xticklabels(), rotation=30)
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(18,12))

#AMT_GOODS_PRICE：Loan annuity
sns.boxplot(x = 'CODE_GENDER', y = 'AMT_GOODS_PRICE', hue = 'TARGET', data = app_train, ax = axis1)
axis1.set_title('CODE_GENDER vs AMT_GOODS_PRICE Comparison')

#DAYS_EMPLOYED How many days before the application the person started current employment
sns.violinplot(x = 'CODE_GENDER', y = 'DAYS_EMPLOYED', hue = 'TARGET', data = app_train, split = True, ax = axis2)
axis2.set_title('CODE_GENDER vs DAYS_EMPLOYED Comparison')

#DAYS_BIRTH How many days before the application did client change his registration
sns.boxplot(x = 'CODE_GENDER', y ='DAYS_BIRTH', hue = 'TARGET', data = app_train, ax = axis3)
axis3.set_title('CODE_GENDER vs DAYS_BIRTH Comparison')

#graph distribution of qualitative data: between two categorical feature
fig, qaxis = plt.subplots(1,3,figsize=(18,12))

sns.barplot(x = 'FLAG_OWN_CAR', y = 'TARGET', hue = 'CODE_GENDER', data=app_train, ax = qaxis[0])
axis1.set_title('FLAG_OWN_CAR vs CODE_GENDER')

sns.barplot(x = 'FLAG_OWN_CAR', y = 'TARGET', hue = 'FLAG_OWN_REALTY', data=app_train, ax  = qaxis[1])
axis1.set_title('FLAG_OWN_CAR vs FLAG_OWN_REALTY')

sns.barplot(x = 'HOUSETYPE_MODE', y = 'TARGET', hue = 'CODE_GENDER', data=app_train, ax  = qaxis[2])
axis1.set_title('HOUSETYPE_MODE vs CODE_GENDER Survival Comparison')
#more side-by-side comparisons
fig, (maxis1, maxis2) = plt.subplots(1, 2,figsize=(18,12))

#how does family size factor with sex & survival compare
sns.pointplot(x="WEEKDAY_APPR_PROCESS_START", y="TARGET", hue="CODE_GENDER", data=app_train,
              palette={"F": "blue", "M": "pink","XNA":"yellow"},
              markers=["*", "o","x"],ax = maxis1)

#how does class factor with sex & survival compare
sns.pointplot(x="NAME_FAMILY_STATUS", y="TARGET", hue="CODE_GENDER", data=app_train,
              palette={"F": "blue", "M": "pink","XNA":"yellow"},
              markers=["*", "o","x"], ax = maxis2)
#copy/Delete part of the modification.
app_train = pd.read_csv('../input/application_train.csv')
app_train1 = app_train
replc = app_train['NAME_FAMILY_STATUS'].unique()
# print(len(replc))
# app_train['NAME_FAMILY_STATUS'].replace('Single / not married','a')
# app_train['NAME_FAMILY_STATUS'].replace('Married','b')
# app_train['NAME_FAMILY_STATUS'].replace('Married','b')

app_train['NFS1'] = app_train['NAME_FAMILY_STATUS']
a = 0
b = ['a','b','c','d','e','f']
for i in replc:
    app_train['NFS1'] = app_train['NFS1'].replace(i,b[a])
    a =a+1
# print(app_train['NAME_FAMILY_STATUS'][1])
app_train['NFS1'].unique()
#facetgrid: https://seaborn.pydata.org/generated/seaborn.FacetGrid.html
e = sns.FacetGrid(app_train, col = 'NFS1')
e.map(sns.pointplot, 'FLAG_OWN_CAR', 'TARGET', 'FLAG_OWN_REALTY', ci=95.0, palette = 'deep')
# e.set(xlim = ['a','b','c','d','e','f'])
e.add_legend()
del app_train
gc.collect()
app_train = app_train1
#plot distributions of number of enquiries to Credit Bureau about the client one hour before application
a = sns.FacetGrid(app_train, hue = 'TARGET', aspect=4 )
a.map(sns.kdeplot, 'AMT_REQ_CREDIT_BUREAU_HOUR', shade= True )
a.set(xlim=(0 , app_train['AMT_REQ_CREDIT_BUREAU_HOUR'].max()))
a.add_legend()
#plot distributions of 
a = sns.FacetGrid(app_train, hue = 'TARGET', aspect=4 )
a.map(sns.kdeplot, 'AMT_INCOME_TOTAL', shade= True )
a.set(xlim=(0 , app_train['AMT_INCOME_TOTAL'].max()))
a.add_legend()
#histogram comparison of 'TARGET', WEEKDAY_APPR_PROCESS_START','CODE_GENDER'
app_train['NCT1'] = app_train['NAME_CONTRACT_TYPE']
app_train['CG1'] = app_train['CODE_GENDER']
h = sns.FacetGrid(app_train, col = 'CG1', row = 'NCT1', hue = 'TARGET')
h.map(plt.hist, 'EXT_SOURCE_1',alpha = 0.5)
h.add_legend()
del app_train
gc.collect()
app_train = app_train1
import random
columns1 = app_train.columns
print(columns)
columns = [i for i in columns1]
#Explore relationships between selecting 8 features 
import random
columns1 = app_train.columns
columns = [i for i in columns1]
col_to_index = {columns[i]:'var'+str(i) for i in range(len(columns))}
corrCols = random.sample(columns,8)
corrCols.append('TARGET')
sampleDf = app_train[corrCols]
for col in corrCols:
    if col != 'TARGET':
        sampleDf.rename(columns = {col:col_to_index[col]},inplace = True)
pp = sns.pairplot(sampleDf, hue = 'TARGET', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )
pp.set(xticklabels=[])
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
    test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    #find if documents in the application_train/test
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    del test_df
    gc.collect()
    return df
# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv('E:/PythonWork/HomeCredit//bureau.csv', nrows=num_rows)
    bb = pd.read_csv('E:/PythonWork/HomeCredit//bureau_balance.csv', nrows=num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg
# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv('E:/PythonWork/HomeCredit/previous_application.csv', nrows=num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Add feature: value ask / value received percentage
    prev['AMT_CREDIT'].fillna(1)
    prev['AMT_CREDIT'].replace(0,1)
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': ['max', 'mean'],
        'AMT_DOWN_PAYMENT': ['max', 'mean'],
        'AMT_GOODS_PRICE': ['max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['max', 'mean'],
        'RATE_DOWN_PAYMENT': ['max', 'mean'],
        'DAYS_DECISION': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg
# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv('E:/PythonWork/HomeCredit/POS_CASH_balance.csv', nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    #根据字段连接
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv('E:/PythonWork/HomeCredit/installments_payments.csv', nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['AMT_INSTALMENT'].fillna(1)
    ins['AMT_INSTALMENT'].replace(0,1)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum', 'min', 'std'],
        'DBD': ['max', 'mean', 'sum', 'min', 'std'],
        'PAYMENT_PERC': ['max', 'mean', 'var', 'min', 'std'],
        'PAYMENT_DIFF': ['max', 'mean', 'var', 'min', 'std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum', 'min', 'std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum', 'std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum', 'std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv('E:/PythonWork/HomeCredit/credit_card_balance.csv', nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg