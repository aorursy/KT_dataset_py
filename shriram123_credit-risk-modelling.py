# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



df = pd.read_csv('../input/loan.csv', low_memory=False)

df.shape
df.head()
df.info()
df.columns

df_description = pd.read_excel('../input/LCDataDictionary.xlsx').dropna()

df_description.style.set_properties(subset=['Description'])
def null_values(df):

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        return mis_val_table_ren_columns
miss_values = null_values(df)

miss_values.head(20)
df.shape
check_null = df.isnull().sum(axis=0).sort_values(ascending=False)/float(len(df))

df.drop(check_null[check_null>0.6].index, axis=1, inplace=True) 

df.dropna(axis=0, thresh=30, inplace=True)
df.head()
df.groupby('loan_status').describe()
dfcheck = df[(df.loan_status == "Charged Off") | (df.loan_status == "Fully Paid")]
dfcheck.groupby('loan_status').describe()
dfcheck.dtypes.value_counts().sort_values().plot(kind='bar')
dfcheck.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
dfcheck['emp_length'].fillna(value=0,inplace=True)



dfcheck['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

sns.countplot(dfcheck.emp_length)
sns.kdeplot(dfcheck.loc[df['loan_status'] == "Charged Off", 'int_rate'], label = 'loan = charged off')

sns.kdeplot(dfcheck.loc[df['loan_status'] == "Fully Paid", 'int_rate'], label = 'loan = fully paid');
fig = plt.figure(figsize=(18,8))

sns.violinplot(x="loan_status",y="int_rate",data=dfcheck, hue="grade")
dfcheck.drop(['emp_title','title','zip_code', 'pymnt_plan'],axis=1,inplace=True)
dfcheck.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
dfcheck.drop(['last_pymnt_d', 'next_pymnt_d', 'sub_grade'],axis=1,inplace=True)

dfcheck['term'] = dfcheck['term'].str.split(' ').str[1]


features = ['loan_amnt', 'term', 'issue_d', 

            'installment', 'grade','emp_length',

            'home_ownership', 'annual_inc','verification_status',

            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 

            'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'initial_list_status', 'last_credit_pull_d', 

            'loan_status', 'earliest_cr_line', 'mths_since_last_delinq', 

            'num_accts_ever_120_pd', 'mort_acc'

           ]

dfmodel= dfcheck[features]

dfmodel['issue_d']= pd.to_datetime(dfmodel['issue_d']).apply(lambda x: int(x.strftime('%Y')))

dfmodel['last_credit_pull_d']= pd.to_datetime(dfmodel['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))

dfmodel['earliest_cr_line']= pd.to_datetime(dfmodel['earliest_cr_line'].fillna('2001-08-01')).apply(lambda x: int(x.strftime('%m')))

dfmodel.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

dfmodel['mths_since_last_delinq'] = dfmodel['mths_since_last_delinq'].fillna(dfmodel['mths_since_last_delinq'].median())
from sklearn import preprocessing



count = 0



for col in dfmodel:

    if dfmodel[col].dtype == 'object':

        if len(list(dfmodel[col].unique())) <= 2:     

            le = preprocessing.LabelEncoder()

            dfmodel[col] = le.fit_transform(dfmodel[col])

            count += 1

            print (col)
dfmodel = pd.get_dummies(dfmodel)

df.dropna(inplace=True)

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
dftarget = dfmodel['loan_status']

dfTrain= dfmodel.drop('loan_status',axis=1)

X_train, X_test, y_train, y_test = train_test_split(dfTrain,dftarget,test_size=0.20,random_state=42)
from lightgbm import LGBMClassifier

model = LGBMClassifier(

            nthread=4,

            n_estimators=10000,

            learning_rate=0.02,

            num_leaves=32,

            colsample_bytree=0.9497036,

            subsample=0.8715623,

            max_depth=8,

            reg_alpha=0.04,

            reg_lambda=0.073,

            min_split_gain=0.0222415,

            min_child_weight=40,

            silent=-1,

            verbose=-1,

            )



model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy_score(y_test,prediction)