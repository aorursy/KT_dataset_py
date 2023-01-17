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
import scipy as sp

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



import datetime as dt



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV,StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



import gc
df_train = pd.read_csv('../input/homework-for-students3/train.csv', index_col=0, parse_dates=['issue_d','earliest_cr_line'])#, skiprows=lambda x: x%20!=0)

df_test = pd.read_csv('../input/homework-for-students3/test.csv', index_col=0, parse_dates=['issue_d','earliest_cr_line'])#, skiprows=lambda x: x%20!=0)
df_train['earliest_cr_line_y'] = df_train.earliest_cr_line.dt.year

df_test['earliest_cr_line_y'] = df_test.earliest_cr_line.dt.year



df_train['cr_line_priod'] = df_train['issue_d'] - df_train['earliest_cr_line']

df_test['cr_line_priod'] = df_test['issue_d'] - df_test['earliest_cr_line']



df_train['cr_line_priod'] = df_train['cr_line_priod'].astype(int)

df_test['cr_line_priod'] = df_test['cr_line_priod'].astype(int)



#df_train['cr_line_priod'] = df_train['cr_line_priod'].apply(np.log1p)

#df_test['cr_line_priod'] = df_test['cr_line_priod'].apply(np.log1p)
df_train.shape
df_train.isnull().sum()
df_train.head()
statelatlong = pd.read_csv('../input/homework-for-students3/statelatlong.csv')



statelatlong.head()
df_train = df_train.merge(statelatlong, left_on='addr_state',right_on='State', how='left')

df_test = df_test.merge(statelatlong, left_on='addr_state',right_on='State', how='left')



df_train.drop(['State'], axis=1, inplace=True)

df_test.drop(['State'], axis=1, inplace=True)



df_train.drop(['Latitude'], axis=1, inplace=True)

df_test.drop(['Latitude'], axis=1, inplace=True)



df_train.drop(['Longitude'], axis=1, inplace=True)

df_test.drop(['Longitude'], axis=1, inplace=True)
df_train['zip_code'].head()
dtypes = {

    'RecordNumber':'int32',

    'Zipcode':'object',

    'ZipCodeType':'object',

    'City':'object',

    'State':'object',

    'LocationType':'object',

    'Lat':'float64',

    'Long':'float64',

    'Xaxis':'float64',

    'Yaxis':'float64',

    'Zaxis':'float64',

    'WorldRegion':'object',

    'Country':'object',

    'LocationText':'object',

    'Location':'object',

    'Decommisioned':'int8',

    'TaxReturnsFiled':'float64',

    'EstimatedPopulation':'float64',

    'TotalWages':'float32',

    'Notes':'object'

}



zipcode_dtl = pd.read_csv('../input/homework-for-students3/free-zipcode-database.csv',index_col=0,dtype=dtypes)



#zipcode_dtl = zipcode_dtl.query('Country.str.contains("US")', engine='python')

zipcode_dtl = zipcode_dtl.query('Decommisioned == 0', engine='python')

zipcode_dtl = zipcode_dtl[zipcode_dtl['Country'] == 'US']

zipcode_dtl = zipcode_dtl[zipcode_dtl['State'] != 'PR']

zipcode_dtl = zipcode_dtl[zipcode_dtl['State'] != 'VI']

zipcode_dtl['zip_code'] = zipcode_dtl['Zipcode'].str[:3]

zipcode_dtl['zip_code'] = zipcode_dtl['zip_code'] + 'xx'



#zipcode_dtl.tail()



zipcode_dtl = zipcode_dtl.groupby(['zip_code']).mean()

zipcode_dtl.drop('Decommisioned', axis=1, inplace=True)

zipcode_dtl.drop('EstimatedPopulation', axis=1, inplace=True)

zipcode_dtl.drop('TaxReturnsFiled', axis=1, inplace=True)

zipcode_dtl.drop('TotalWages', axis=1, inplace=True)
zipcode_dtl.head(20)
df_train['zip_code'].head()
df_train = df_train.merge(zipcode_dtl, on='zip_code', how='left')

df_test = df_test.merge(zipcode_dtl, on='zip_code', how='left')
df_train.head()
df_test.head()
df_train[df_train.loan_condition==1].loan_amnt.mean()
df_train[df_train.loan_condition==0].loan_amnt.mean()
df_train.describe()
df_test.describe()
df_train.dtypes
df_train.isnull().sum()
df_test.isnull().sum()
df_train.grade.value_counts() / len(df_train)
df_test.grade.value_counts() / len(df_test)
df_train[df_train.loan_condition ==1].grade.value_counts()
df_train[df_train.loan_condition ==0].grade.value_counts()
df_train.sub_grade.value_counts()
df_train.dtypes
df_train['annual_inc'].head()
f = 'annual_inc'



plt.figure(figsize=[7,7])

df_train[f].hist(density=False, alpha=0.5, bins=10)

df_test[f].hist(density=False, alpha=0.5, bins=10)

plt.xlabel(f)

plt.ylabel('count')

plt.show()
f = 'purpose'



df_train[f].value_counts() / len(df_train)
df_test[f].value_counts() / len(df_test)
#df_train = df_train.query('issue_d.str.contains("2015")', engine='python')

#issue_year = df_train.issue_d.dt.year



df_train = df_train[df_train.issue_d.dt.year == 2015]



df_train.head()
df_train.shape
df_train.isnull().sum()
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
X_train.shape
X_train.describe()
X_train.columns
usgdp = pd.read_csv('../input/homework-for-students3/US_GDP_by_State.csv')

#usgdp.drop(['Gross State Product'], axis=1, inplace=True)

#usgdp.drop(['State & Local Spending'], axis=1, inplace=True)

#usgdp.drop(['Population (million)'], axis=1, inplace=True)
usgdp.columns
# 2013年データ.drop(

usgdp2013 = usgdp.query('year == "2013"')

# 2014年データ, axis=1, inplace=True)

usgdp2014 = usgdp.query('year == "2014"')

# 2015年データ

usgdp2015 = usgdp.query('year == "2015"')



# 学習データには2013年と2014年（過去2年分）を追加

X_train = X_train.merge(usgdp2013, left_on='City', right_on='State', how='left')

X_train['Gross State Product 2y'] = X_train['Gross State Product']

X_train['State & Local Spending 2y'] = X_train['State & Local Spending']

X_train['Population (million) 2y'] = X_train['Population (million)']

X_train['Real State Growth % 2y'] = X_train['Real State Growth %']

X_train.drop(['Gross State Product'], axis=1, inplace=True)

X_train.drop(['State & Local Spending'], axis=1, inplace=True)

X_train.drop(['Population (million)'], axis=1, inplace=True)

X_train.drop(['Real State Growth %'], axis=1, inplace=True)

X_train.drop(['State'], axis=1, inplace=True)

X_train.drop(['year'], axis=1, inplace=True)



X_train = X_train.merge(usgdp2014, left_on='City', right_on='State', how='left')

X_train['Gross State Product 1y'] = X_train['Gross State Product']

X_train['State & Local Spending 1y'] = X_train['State & Local Spending']

X_train['Population (million) 1y'] = X_train['Population (million)']

X_train['Real State Growth % 1y'] = X_train['Real State Growth %']

X_train.drop(['Gross State Product'], axis=1, inplace=True)

X_train.drop(['State & Local Spending'], axis=1, inplace=True)

X_train.drop(['Population (million)'], axis=1, inplace=True)

X_train.drop(['Real State Growth %'], axis=1, inplace=True)

X_train.drop(['City'], axis=1, inplace=True)

X_train.drop(['State'], axis=1, inplace=True)

X_train.drop(['year'], axis=1, inplace=True)



# テストデータには2014年と2015年（過去2年分）を追加

X_test = X_test.merge(usgdp2014, left_on='City', right_on='State', how='left')

X_test['Gross State Product 2y'] = X_test['Gross State Product']

X_test['State & Local Spending 2y'] = X_test['State & Local Spending']

X_test['Population (million) 2y'] = X_test['Population (million)']

X_test['Real State Growth % 2y'] = X_test['Real State Growth %']

X_test.drop(['Gross State Product'], axis=1, inplace=True)

X_test.drop(['State & Local Spending'], axis=1, inplace=True)

X_test.drop(['Population (million)'], axis=1, inplace=True)

X_test.drop(['Real State Growth %'], axis=1, inplace=True)

X_test.drop(['State'], axis=1, inplace=True)

X_test.drop(['year'], axis=1, inplace=True)



X_test = X_test.merge(usgdp2015, left_on='City', right_on='State', how='left')

X_test['Gross State Product 1y'] = X_test['Gross State Product']

X_test['State & Local Spending 1y'] = X_test['State & Local Spending']

X_test['Population (million) 1y'] = X_test['Population (million)']

X_test['Real State Growth % 1y'] = X_test['Real State Growth %']

X_test.drop(['Gross State Product'], axis=1, inplace=True)

X_test.drop(['State & Local Spending'], axis=1, inplace=True)

X_test.drop(['Population (million)'], axis=1, inplace=True)

X_test.drop(['Real State Growth %'], axis=1, inplace=True)

X_test.drop(['City'], axis=1, inplace=True)

X_test.drop(['State'], axis=1, inplace=True)

X_test.drop(['year'], axis=1, inplace=True)



X_test.head()
X_train.columns
#X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

#X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)
cats = []



for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
X_train['emp_title'].head(10)
col = 'purpose'



encoder = OneHotEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.fit_transform(X_test[col].values)
enc_train.head()
enc_test.head()
encoder = OrdinalEncoder()

enc_train = encoder.fit_transform(X_train[col].values)

enc_test = encoder.fit_transform(X_test[col].values)
enc_train.head()
enc_test.head()
summary = X_train[col].value_counts()

summary
enc_train = X_train[col].map(summary)

enc_test = X_test[col].map(summary)
enc_train.head()
enc_test.head()
target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



summary = X_temp.groupby([col])[target].mean()

enc_test = X_test[col].map(summary)



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



enc_train = Series(np.zeros(len(X_train)), index=X_train.index)



for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

    X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

    X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

    

    summary = X_train_.groupby([col])[target].mean()

    enc_train.iloc[val_ix] = X_val[col].map(summary)
enc_train
enc_test
TXT_train = X_train.emp_title.copy()

TXT_test = X_test.emp_title.copy()



#cats.remove('emp_title')



cats
#grade, sub_grade, emp_lengthを辞書に基づいてLabel Encoding



mappingdict = {

    "grade": { "A": 7,"B": 6,"C": 5,"D": 4,"E": 3,"F": 2,"G": 1 },

    "sub_grade": {

        "A1": 45,"A2": 44,"A3": 43,"A4": 42,"A5": 41,

        "B1": 40,"B2": 39,"B3": 38,"B4": 37,"B5": 36,

        "C1": 35,"C2": 34,"C3": 33,"C4": 32,"C5": 31,

        "D1": 30,"D2": 29,"D3": 28,"D4": 27,"D5": 26,

        "E1": 25,"E2": 24,"E3": 23,"E4": 22,"E5": 21,

        "F1": 20,"F2": 19,"F3": 18,"F4": 17,"F5": 16,

        "G1": 15,"G2": 14,"G3": 13,"G4": 12,"G5": 11

    }

    ,"emp_length": {

        "10+ years": 10,

        "9 years": 9,

        "8 years": 8,

        "7 years": 7,

        "6 years": 6,

        "5 years": 5,

        "4 years": 4,

        "3 years": 3,

        "2 years": 2,

        "1 year": 1,

        "< 1 year": 0,

        "n/a": ""

    }

}

X_train = X_train.replace(mappingdict)

X_test = X_test.replace(mappingdict)

mappingcol = ['grade','sub_grade','emp_length']

X_train[mappingcol] = X_train[mappingcol].astype(float)

X_test[mappingcol] = X_test[mappingcol].astype(float)
X_train.head()
X_test.head()
X_train.drop(['issue_d'], axis=1, inplace=True)

X_test.drop(['issue_d'], axis=1, inplace=True)



X_train.drop(['earliest_cr_line'], axis=1, inplace=True)

X_test.drop(['earliest_cr_line'], axis=1, inplace=True)



#X_train.drop(['title'], axis=1, inplace=True)

#X_test.drop(['title'], axis=1, inplace=True)



#X_train.drop(['grade'], axis=1, inplace=True)

#X_test.drop(['grade'], axis=1, inplace=True)
cats.remove('grade')

cats.remove('sub_grade')

#cats.remove('emp_length')

#cats.remove('title')
cats
X_train.isnull().sum()
oe = OrdinalEncoder(cols=cats, return_df=False)
X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
X_train.head()
X_test.head()
# 欠損の確認

X_train.isnull().sum()



X_train['null_count'] = X_train.isnull().sum(axis=1)

X_test['null_count'] = X_test.isnull().sum(axis=1)
X_train.head()
#X_all = pd.concat([X_train,X_test], axis=0)



nullcheck = X_train.columns[X_train.isnull().sum() != 0].values



for col in nullcheck:

    X_train[col + '_null'] = 0

    X_train[col + '_null'] = X_train[col].isnull()

    X_test[col + '_null'] = 0

    X_test[col + '_null'] =  X_test[col].isnull()



X_train.head()

X_train.isnull().sum()
X_test.isnull().sum()
#X_train['mths_since_last_delinq'].fillna(9999, inplace=True)

#X_train['mths_since_last_record'].fillna(9999, inplace=True)

#X_train['mths_since_last_major_derog'].fillna(9999, inplace=True)



#X_test['mths_since_last_delinq'].fillna(9999, inplace=True)

#X_test['mths_since_last_record'].fillna(9999, inplace=True)

#X_test['mths_since_last_major_derog'].fillna(9999, inplace=True)



#X_train.fillna(X_train.median(), inplace=True)

#X_test.fillna(X_train.median(), inplace=True)



X_train.fillna(-9999, inplace=True)

X_test.fillna(-9999, inplace=True)
X_train.isnull().sum()
cats
cats.append('grade')

cats.append('sub_grade')

cats.append('acc_now_delinq')

cats.append('mths_since_last_delinq')

cats.append('tot_coll_amt')



cats
target = 'loan_condition'

X_temp = pd.concat([X_train, y_train], axis=1)



for col in cats:

    summary = X_temp.groupby([col])[target].mean()

    enc_test = X_test[col].map(summary)

    

    X_test[col + '_te'] = enc_test

    

    skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)

    

    enc_train = Series(np.zeros(len(X_train)), index=X_train.index)

    

    for i, (train_ix, val_ix) in enumerate((skf.split(X_train, y_train))):

        X_train_, _ = X_temp.iloc[train_ix], y_train.iloc[train_ix]

        X_val, _ = X_temp.iloc[val_ix], y_train.iloc[val_ix]

        

        summary = X_train_.groupby([col])[target].mean()

        enc_train.iloc[val_ix] = X_val[col].map(summary)

        

        X_train[col + '_te'] = enc_train



X_train.head()
X_test.head()
X_train['loan_amnt_sg'] = X_train['loan_amnt'] * X_train['sub_grade']

X_train['annual_inc_sg'] = X_train['annual_inc'] * X_train['sub_grade']

X_train['installment_sg'] = X_train['installment'] * X_train['sub_grade']

X_train['dti_sg'] = X_train['dti'] * X_train['sub_grade']

X_train['loan_inc'] = X_train['loan_amnt'] * X_train['annual_inc']

X_train['loan_installment'] = X_train['loan_amnt'] * X_train['installment']

X_train['loan_dti'] = X_train['loan_amnt'] * X_train['dti']

X_train['inc_installment'] = X_train['annual_inc'] * X_train['installment']

X_train['inc_dti'] = X_train['annual_inc'] * X_train['dti']

X_train['installment_dti'] = X_train['installment'] * X_train['dti']



X_test['loan_amnt_sg'] = X_test['loan_amnt'] * X_test['sub_grade']

X_test['annual_inc_sg'] = X_test['annual_inc'] * X_test['sub_grade']

X_test['installment_sg'] = X_test['installment'] * X_test['sub_grade']

X_test['dti_sg'] = X_test['dti'] * X_test['sub_grade']

X_test['loan_inc'] = X_test['loan_amnt'] * X_test['annual_inc']

X_test['loan_installment'] = X_test['loan_amnt'] * X_test['installment']

X_test['loan_dti'] = X_test['loan_amnt'] * X_test['dti']

X_test['inc_installment'] = X_test['annual_inc'] * X_test['installment']

X_test['inc_dti'] = X_test['annual_inc'] * X_test['dti']

X_test['installment_dti'] = X_test['installment'] * X_test['dti']



X_train['revol_amnt'] = X_train['revol_bal'] / (101 - X_train['revol_util'])

X_test['revol_amnt'] = X_test['revol_bal'] / (101 - X_test['revol_util'])



X_train['annual_inc_sg'] = X_train['annual_inc_sg'] + 1

X_test['annual_inc_sg'] = X_test['annual_inc_sg'] + 1



X_train['loan_amnt_sg_rate'] = round(X_train['loan_amnt_sg'] / X_train['annual_inc_sg'], 6)

X_test['loan_amnt_sg_rate'] = round(X_test['loan_amnt_sg'] / X_test['annual_inc_sg'], 6)



X_train['inst_inc_rate'] = round((X_train['installment'] * 12) / X_train['annual_inc_sg'], 6)

X_test['inst_inc_rate'] = round((X_test['installment'] * 12) / X_test['annual_inc_sg'], 6)



X_train['inst_loan_rate'] = round(X_train['installment'] / X_train['loan_amnt_sg'], 6)

X_test['inst_loan_rate'] = round(X_test['installment'] / X_test['loan_amnt_sg'], 6)



X_train['dti_sg_rate'] = round(X_train['dti_sg'] / X_train['annual_inc_sg'], 6)

X_test['dti_sg_rate'] = round(X_test['dti_sg'] / X_test['annual_inc_sg'], 6)



X_train['acc_rate'] = round(X_train['open_acc'] / X_train['total_acc'], 6)

X_test['acc_rate'] = round(X_test['open_acc'] / X_test['total_acc'], 6)



X_train['Gross State Product diff'] = X_train['Gross State Product 2y'] - X_train['Gross State Product 1y']

X_train['State & Local Spending diff'] = X_train['State & Local Spending 2y'] - X_train['State & Local Spending 1y']

X_train['Population (million) diff'] = X_train['Population (million) 2y'] - X_train['Population (million) 1y']

X_train['Real State Growth % diff'] = X_train['Real State Growth % 2y'] - X_train['Real State Growth % 1y']



X_train['Gross State Product sum'] = X_train['Gross State Product 2y'] + X_train['Gross State Product 1y']

X_train['State & Local Spending sum'] = X_train['State & Local Spending 2y'] + X_train['State & Local Spending 1y']

X_train['Population (million)  grow rate'] = X_train['Population (million) 2y'] / X_train['Population (million) 1y']

X_train['Real State Growth % x'] = X_train['Real State Growth % 2y'] * X_train['Real State Growth % 1y']





X_test['Gross State Product diff'] = X_test['Gross State Product 2y'] - X_test['Gross State Product 1y']

X_test['State & Local Spending diff'] = X_test['State & Local Spending 2y'] - X_test['State & Local Spending 1y']

X_test['Population (million) diff'] = X_test['Population (million) 2y'] - X_test['Population (million) 1y']

X_test['Real State Growth % diff'] = X_test['Real State Growth % 2y'] - X_test['Real State Growth % 1y']



X_test['Gross State Product sum'] = X_test['Gross State Product 2y'] + X_test['Gross State Product 1y']

X_test['Gross State Product sum'] = X_test['Gross State Product 2y'] + X_test['Gross State Product 1y']

X_test['Population (million) grow rate'] = X_test['Population (million) 2y'] / X_test['Population (million) 1y']

X_test['Real State Growth % x'] = X_test['Real State Growth % 2y'] * X_test['Real State Growth % 1y']

#X_train.drop(['emp_title'], axis=1, inplace=True)

#X_test.drop(['emp_title'], axis=1, inplace=True)

#X_train.drop(['addr_state'], axis=1, inplace=True)

#X_test.drop(['addr_state'], axis=1, inplace=True)

#X_train.drop(['purpose'], axis=1, inplace=True)

#X_test.drop(['purpose'], axis=1, inplace=True)

#X_train.drop(['initial_list_status'], axis=1, inplace=True)

#X_test.drop(['initial_list_status'], axis=1, inplace=True)

#X_train.drop(['application_type'], axis=1, inplace=True)

#X_test.drop(['application_type'], axis=1, inplace=True)
cats = X_train.columns



X_all = pd.concat([X_train,X_test], axis=0, sort=True)



for col in cats:

    freq = X_all[col].value_counts()

    col_count = col + '_count'

    X_all[col_count] = X_all[col].map(freq)



#freq = X_all['grade'].value_counts()

#X_all['grade_count'] = X_all['grade'].map(freq)



#freq = X_all['sub_grade'].value_counts()

#X_all['sub_grade_count'] = X_all['sub_grade'].map(freq)

    

freq = X_all['earliest_cr_line_y'].value_counts()

X_all['earliest_cr_line_y_count'] = X_all['earliest_cr_line_y'].map(freq)



#X_all.drop(['earliest_cr_line_y'], axis=1, inplace=True)

#X_train.drop(['earliest_cr_line_y'], axis=1, inplace=True)

#X_test.drop(['earliest_cr_line_y'], axis=1, inplace=True)



#X_all.drop(['home_ownership'], axis=1, inplace=True)

#X_train.drop(['home_ownership'], axis=1, inplace=True)

#X_test.drop(['home_ownership'], axis=1, inplace=True)



X_train = X_all.iloc[:X_train.shape[0],:]

X_test = X_all.iloc[X_train.shape[0]:,:]



del X_all
X_train.head()
LGBMClassifier()
X_train.columns
scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier()

    #clf = LGBMClassifier(boosting_type='dart', n_estimators=1000)

    clf = LGBMClassifier()

    

    

    #clf.fit(X_train_, y_train_)

    clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred_1 = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred_1)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
clf.fit(X_train, y_train)



y_pred_1 = clf.predict_proba(X_test)[:,1]
scores = []



skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)



for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

    X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

    X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

    #clf = GradientBoostingClassifier()

    clf = LGBMClassifier(boosting_type='dart', n_estimators=1000)

    #clf = LGBMClassifier()

    

    

    #clf.fit(X_train_, y_train_)

    clf.fit(X_train_, y_train_, eval_metric='auc', eval_set=[(X_val, y_val)])

    y_pred_2 = clf.predict_proba(X_val)[:,1]

    score = roc_auc_score(y_val, y_pred_2)

    scores.append(score)

    

    print('CV Score of Fold_%d is %f' % (i, score))
print(np.mean(scores))

print(scores)
clf.fit(X_train, y_train)



y_pred_2 = clf.predict_proba(X_test)[:,1]
y_pred = (y_pred_1 + y_pred_2) / 2
fig, ax = plt.subplots(figsize=(10, 15))

lgb.plot_importance(clf, max_num_features=50, ax=ax, importance_type='gain')
submission = pd.read_csv('../input/homework-for-students3/sample_submission.csv', index_col=0)#, skiprows=lambda x:x%20!=0)



submission.loan_condition = y_pred

submission.to_csv('submission.csv')
#TXT_train.fillna('#', inplace=True)

#TXT_test.fillna('#', inplace=True)
#tfidf = TfidfVectorizer(max_features=1000, use_idf=True)
#TXT_train = tfidf.fit_transform(TXT_train)

#TXT_test = tfidf.transform(TXT_test)
#TXT_train
#TXT_train.shape
#TXT_train.todense()
#sp.sparse.hstack([X_train.values, TXT_train])
#sp.sparse.hstack([X_train.values, TXT_train]).todense()
del X_train

del y_train

del X_test