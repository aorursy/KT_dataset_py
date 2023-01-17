# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Lending Club Case Study Data Dictionary

# https://upgradstorage.s3.ap-south-1.amazonaws.com/LendingCaseStudy/Data_Dictionary.xlsx



# Lending Club Case Study DataSet

# https://upgradstorage.s3.ap-south-1.amazonaws.com/LendingCaseStudy/loan.csv
loan_data = pd.read_csv("https://upgradstorage.s3.ap-south-1.amazonaws.com/LendingCaseStudy/loan.csv")
loan_data.head()
loan_data.shape
pd.set_option('display.max_colwidth', -1)

pd.set_option("display.max_rows", 40)
loan_data.info(verbose=True, null_counts=True)
loan_data.describe(include='all').loc['unique', :]
loan_data['collections_12_mths_ex_med'].unique()
loan_data['policy_code'].unique()
loan_data['dti_joint'].unique()
loan_data['term'].unique()
loan_data['pymnt_plan'].unique()
loan_data['initial_list_status'].unique()
loan_data['loan_amnt'].describe()
loan_data['funded_amnt'].describe()
loan_data['int_rate'].describe()
loan_data['initial_list_status'].unique()
loan_data['loan_status'].unique()
loan_data.dropna(how='all', inplace=True, axis='columns')

loan_data.shape
loan_data.drop(columns=['policy_code', 'application_type', 'initial_list_status'], inplace=True)

loan_data.shape
print(loan_data.tax_liens.unique())

loan_data.drop(columns=['tax_liens', 'chargeoff_within_12_mths'], inplace=True)

loan_data.shape
print(loan_data.collections_12_mths_ex_med.unique())

loan_data.drop(columns=['collections_12_mths_ex_med'], inplace=True)

loan_data.shape
loan_data['defaulted'] = loan_data['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

loan_data.head()
total = loan_data.isnull().sum().sort_values(ascending=False)

percent = (loan_data.isnull().sum()/loan_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
# There are 3 columns with most of them are null values which shown above can be removed.

loan_data.drop(columns=['next_pymnt_d', 'mths_since_last_record', 'mths_since_last_delinq'], inplace=True)

loan_data.shape
for k in loan_data:

    print(k)

    print('-----------')

    print(loan_data[k].describe())

    print('-----------')
# ====> From the above analysis we can get that there are columns with simply "0" as the value and can't have any effect

# ====> Can be removed from the dataset

print(loan_data['delinq_amnt'].unique())

loan_data.drop(columns=['acc_now_delinq'], inplace=True)

loan_data.drop(columns=['delinq_amnt'], inplace=True)

loan_data.shape
loan_data.drop(columns=['pymnt_plan'], inplace=True)

loan_data.shape
# ====> Remove columns

loan_data.columns
sns.distplot(loan_data.loan_amnt, bins = 20)
loan_data.groupby('loan_status')['loan_status'].count().plot.bar()
print(loan_data.groupby('grade').grade.count())

loan_data.groupby('grade').grade.count().plot.bar()
loan_data.groupby('home_ownership').home_ownership.count()

sns.countplot(loan_data.home_ownership, order=loan_data.home_ownership.value_counts().index)
# ===> funded ammout, loan amount, funded amount investment distplot

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)

sns.distplot(loan_data['funded_amnt'])



plt.subplot(2, 3, 2)

sns.distplot(loan_data['loan_amnt'])



plt.subplot(2, 3, 3)

sns.distplot(loan_data['funded_amnt_inv'])
loan_data.pivot_table(values = 'loan_amnt', index = ['grade', 'loan_status'], aggfunc = ['sum'])
# ===> Defaulted vs FullyPaid/Currently paying customers 

plt.figure(figsize=(10,10))

plt.subplot(2, 2, 1)

loan_data.groupby('loan_status').loan_status.count().plot.bar()

plt.subplot(2, 2, 2)

loan_data.groupby('defaulted')['defaulted'].count().plot.bar()

plt.show()
loan_data['defaulted'].describe()
plt.figure(figsize=(5, 5))

order_grades = loan_data.groupby(["grade"])['defaulted'].aggregate(np.mean).reset_index().sort_values('defaulted')

sns.barplot(data=loan_data, y='defaulted', x='grade', order=order_grades['grade'])

plt.show()
plt.figure(figsize=(12, 8))

order_sub_grades = loan_data.groupby(["sub_grade"])['defaulted'].aggregate(np.mean).reset_index().sort_values('defaulted')

sns.barplot(data=loan_data, y='defaulted', x='sub_grade', order=order_sub_grades['sub_grade'])

plt.show()
plt.figure(figsize=(25, 8))

order_business = loan_data.groupby('purpose').defaulted.aggregate(np.mean).reset_index().sort_values('defaulted')

sns.barplot(data=loan_data, y='defaulted', x='purpose', order=order_business['purpose'])

plt.show()



plt.figure(figsize=(25, 8))

sns.countplot(data=loan_data, x='purpose', order=order_business['purpose'])

plt.show()
loan_data.issue_d = pd.to_datetime(loan_data['issue_d'], format='%b-%y')

loan_data.loc[:, 'issue_year'] = loan_data.issue_d.dt.year

loan_data.loc[:, 'issue_month'] = loan_data.issue_d.dt.month

loan_data.head(3)
plt.figure(figsize=(10, 10))

sns.barplot(data=loan_data, y='defaulted', x='issue_year')

plt.show()
loan_data.int_rate.replace('%', '', inplace=True, regex=True)

loan_data.int_rate =loan_data.int_rate.astype('float64')

loan_data.int_rate.head()
# =====> defaulted users w.r.t to issue year and verification status

plt.figure(figsize=(10, 5))

sns.barplot(data=loan_data, y='defaulted', x='issue_year',hue='verification_status')

plt.show()
plt.figure(figsize=(14, 14))



_columns = ['defaulted', 'loan_amnt', 'int_rate', 'grade', 'sub_grade', 'loan_status', 'issue_year', 'open_acc']

sns.pairplot(loan_data[_columns], diag_kind='kde')

plt.show()
def plot_range_with_defaults(_data, xcolumn, _bins=10, _width=10, _height=5, ycolumn='defaulted', action='cut'):

    plt.figure(figsize=(_width, _height))

    if (action == 'cut'):

        _ranges = pd.cut(_data[xcolumn], _bins)

    else:

        _ranges = pd.qcut(_data[xcolumn], _bins)

    _a = pd.DataFrame({ xcolumn : _ranges, ycolumn : _data[ycolumn]})

    sns.barplot(y=_a[ycolumn], x=_a[xcolumn])

    plt.show()
plot_range_with_defaults(loan_data, 'int_rate', 5)
#=====> 60 months term

long_term_loan = loan_data[loan_data.term ==' 60 months']

plot_range_with_defaults(long_term_loan, 'int_rate', 5)
#=====> 36 months term

short_term_loan = loan_data[loan_data.term ==' 36 months']

plot_range_with_defaults(short_term_loan, 'int_rate', 5)
plot_range_with_defaults(loan_data, 'annual_inc', 10, 20, 8, 'defaulted', 'qcut')
plot_range_with_defaults(loan_data, 'int_rate', 5)
plot_range_with_defaults(loan_data, 'open_acc', 25, 35, 10)
loan_data['emp_experience'] = loan_data['emp_length'].astype('str') 

loan_data['emp_experience'] = loan_data['emp_experience'].apply(lambda x: x.split(' ')[0]).replace({'<': 0,'10+': 10},regex=True)

loan_data['emp_experience'] = loan_data['emp_experience'].astype('float64')

loan_data.emp_experience.head()
plot_range_with_defaults(loan_data, 'emp_experience', 10, 15, 5, 'defaulted')