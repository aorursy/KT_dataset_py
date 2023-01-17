# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/loan.csv")
df.head()
df.shape
df.describe()
df.info()
df.isnull().sum()
sns.countplot(x='grade',data=df, palette='bwr')
plt.figure(figsize=(20,5))

sns.countplot(x='loan_status',data=df, palette='bwr')
#plt.figure(figsize=(20,5))

sns.countplot(x='home_ownership',data=df, palette='bwr')
sns.countplot(x='term',data=df, palette='bwr')
fig, ax = plt.subplots(1, 2, figsize=(16,5))

sns.distplot(df['loan_amnt'], ax=ax[0])

ax[0].set_title("Loan Amount Distribution")

sns.distplot(df['funded_amnt'], ax=ax[1])

ax[1].set_title("Funded Amount Distribution")
plt.figure(figsize=(25,10))

sns.countplot(x='purpose',data=df, palette='bwr')
print(df['purpose'].value_counts())
plt.figure(figsize=(5,10))

sns.boxplot(x=df['term'], y=df['loan_amnt'])
plt.figure(figsize=(5,10))

sns.boxplot(x=df['verification_status'], y=df['loan_amnt'])
def getmonth(month):

    return month.split("-")[0]



def getyear(year):

    return year.split("-")[1]



df['Month'] = df.issue_d.apply(getmonth)

df['Year'] = df.issue_d.apply(getyear)
sns.pointplot(x=df['Month'], y=df['loan_amnt'],  data=df)
sns.countplot(x=df['Month'],  data=df)
sns.pointplot(x=df['Year'], y=df['loan_amnt'],  data=df)
sns.countplot(x=df['Year'], data=df)
# Hot encode some categorical features 

columns = ['emp_length','home_ownership']



for col in columns:

    tmp_df = pd.get_dummies(df[col], prefix=col)

    df = pd.concat((df, tmp_df), axis=1)
df.drop(['id',

         'member_id',

         'loan_status',

           'term',

           'grade',

           'sub_grade',

           'emp_title',

           'home_ownership',

           'verification_status',

           'issue_d',

           'url',

           'desc',

           'title',

           'zip_code',

           'addr_state',

           'emp_length',

           'earliest_cr_line',

           'addr_state',

           'initial_list_status',

           'pymnt_plan',

           'next_pymnt_d',

           'last_credit_pull_d',

           'purpose',

           'verification_status_joint',

           'application_type'], axis=1, inplace=True)
df.isnull().sum()
df.drop(['annual_inc_joint','dti_joint','acc_now_delinq','tot_coll_amt','tot_cur_bal','open_acc_6m','open_il_6m',

         'open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc',

         'all_util','total_rev_hi_lim','inq_fi','total_cu_tl','inq_last_12m'], axis=1, inplace=True)
df.drop(['mths_since_last_major_derog','mths_since_last_record','mths_since_last_delinq'], axis=1, inplace=True)
df.drop(['last_pymnt_d'], axis=1, inplace=True)
df.fillna(0, inplace=True)
X = df[['loan_amnt', 'funded_amnt',

'funded_amnt_inv',

'installment',

'annual_inc',

'dti',

'delinq_2yrs',

'inq_last_6mths',

'open_acc',

'pub_rec',

'revol_bal',

'revol_util',

'total_acc',

'out_prncp',

'out_prncp_inv',

'total_pymnt',

'total_pymnt_inv',

'total_rec_prncp',

'total_rec_int',

'total_rec_late_fee',

'recoveries',

'collection_recovery_fee',

'last_pymnt_amnt',

'collections_12_mths_ex_med',

'policy_code',

'emp_length_1 year',

'emp_length_10+ years',

'emp_length_2 years',

'emp_length_3 years',

'emp_length_4 years',

'emp_length_5 years',

'emp_length_6 years',

'emp_length_7 years',

'emp_length_8 years',

'emp_length_9 years',

'emp_length_< 1 year',

'home_ownership_ANY',

'home_ownership_MORTGAGE',

'home_ownership_NONE',

'home_ownership_OTHER',

'home_ownership_OWN',

'home_ownership_RENT']]

X.head()
y=df[['int_rate']]

y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr = lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)
y_pred = lr.predict([[5000,5000,4975,162.87,24000,27.65,0,1,3,0,13648,83.7,9,0,0,5861.071414,5831.78,5000,861.07,0,0,0,171.62,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
y_pred
f, ax = plt.subplots(figsize=(10,8))

corr = X.corr()

sns.heatmap(corr,

           xticklabels = corr.columns.values,

           yticklabels = corr.columns.values)