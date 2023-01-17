# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas packages
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Changing the value in permalink to lower case
loan = pd.read_csv('../input/loan.csv')
loan.head()
loan.info()
loan.describe()
#Inspecting the Null values
loan.isnull().sum(axis=0)
round(100*(loan.isnull().sum()/len(loan.index)),2)
filtered_loandata = loan.dropna(axis='columns', how='all')
round(100*(filtered_loandata.isnull().sum()/len(filtered_loandata.index)),2)
filtered_loandata = filtered_loandata.drop(["next_pymnt_d", "mths_since_last_record", "mths_since_last_delinq"], axis = 1 )
filtered_loandata.info()
filtered_loandata.policy_code.unique()
filtered_loandata.term.unique()
plt.figure(1)
plt.subplot(321) # bar plot with the number of loans for each category of home ownership
filtered_loandata.home_ownership.value_counts().plot.bar(figsize=(20,15), title= 'Home Ownership')

plt.subplot(322) # bar plot with the number of borrowers within each category of loan purpose
filtered_loandata.purpose.value_counts().plot.bar(title= 'Loan Purpose')

plt.subplot(323)
filtered_loandata.loan_status.value_counts().plot.bar(title= 'Status of the loan')

plt.subplot(324)
filtered_loandata.grade.value_counts().plot.bar(title= 'LC assigned loan grade')

plt.subplot(324)
filtered_loandata.grade.value_counts().plot.bar(title= 'LC assigned loan grade')

plt.subplot(325)
filtered_loandata.emp_length.value_counts().plot.bar(title= 'Employee length')

plt.subplot(326)
filtered_loandata.term.value_counts().plot.bar(title= 'Loan term')

plt.show()
# let's check the amount of missing data
filtered_loandata.isnull().mean()
# let's inspect the variable emp_length
filtered_loandata.emp_length.unique()
filtered_loandata['emp_length'].fillna('0 years', inplace = True)
# let's inspect the variable emp_length
filtered_loandata.emp_length.unique()
# let's look at the percentage of borrowers within
# each label / category of the emp_length variable

filtered_loandata.emp_length.value_counts() / len(filtered_loandata)
filtered_loandata.loc[filtered_loandata.emp_length.isnull()]
# the variable emp_length has many categories. I will summarise it
# into 3 for simplicity:'0-10 years' or '10+ years' or 'n/a'

# let's build a dictionary to re-map emp_length to just 3 categories:

length_dict = {k:'0-10 years' for k in filtered_loandata.emp_length.unique()}
length_dict['10+ years']='10+ years'
length_dict['0 years']='n/a'

# let's look at the dictionary
length_dict
# let's re-map the emp_length

filtered_loandata['emp_length_redefined'] = filtered_loandata.emp_length.map(length_dict)
filtered_loandata.emp_length_redefined.unique()
# let's calculate the proportion of working years
# with same employer for those who miss data on employer name

# number of borrowers for whom employer name is missing
value = len(filtered_loandata[filtered_loandata.emp_title.isnull()])

# % of borrowers for whom employer name is missing 
# within each category of employment length
filtered_loandata[filtered_loandata.emp_title.isnull()].groupby(['emp_length_redefined'])['emp_length'].count().sort_values() / value
# let's do the same for those bororwers who reported
# the employer name

# number of borrowers for whom employer name is present
value = len(filtered_loandata.dropna(subset=['emp_title']))

# % of borrowers within each category
filtered_loandata.dropna(subset=['emp_title']).groupby(['emp_length_redefined'])['emp_length'].count().sort_values() / value
# modification of the data of the int_rate and changing it into float datatype
#filtered_loandata['int_rate'] = filtered_loandata['int_rate'].str.rstrip('%').astype('float')

# list of Object type columns with there one data
object_column_loan = filtered_loandata.select_dtypes(include=['object'])
print(object_column_loan.iloc[0])
cols = ['grade','home_ownership','verification_status','loan_status','pymnt_plan','purpose', 'addr_state','initial_list_status','last_credit_pull_d','application_type']
for name in cols:
    print(name,':')
    print(object_column_loan[name].value_counts(),'\n')
# Drop unneccesary columns as it has only single values
filtered_loandata = filtered_loandata.drop(['pymnt_plan', 'initial_list_status', 'application_type','policy_code','collections_12_mths_ex_med','acc_now_delinq'], axis=1)
print(filtered_loandata.shape)
#Inspecting the Null values
filtered_loandata.isnull().sum()
filtered_loandata.info()
cols = ['last_credit_pull_d','earliest_cr_line']
for name in cols:
    print(name,':')
    print(object_column_loan[name].value_counts(),'\n')
filtered_loandata.revol_util.value_counts()
#Changing the datatype of the emp_length to int
filtered_loandata["revol_util"] = filtered_loandata["revol_util"].astype('float')
#Seeing the revol_util statistics
filtered_loandata["revol_util"].describe()
loan_by_grade = filtered_loandata.groupby("grade").mean()
avg_loan = loan_by_grade['loan_amnt'].reset_index()
plt.subplots(figsize=(10,6))
sns.barplot(x='grade', y='loan_amnt', data=avg_loan)
filtered_loandata['loan_status'].unique()
filtered_loandata['loan_status_numeric'] = np.where(filtered_loandata['loan_status'] == 'Charged Off', 1, 0)
filtered_loandata[['loan_status', 'loan_status_numeric']].head()
plt.subplots(figsize=(12,6))
sns.violinplot(x="grade", y="int_rate", hue='loan_status_numeric', split=True, inner="quart", data=filtered_loandata, order="ABCDEFG")
sns.despine(left=True)
# let's see how much money Lending Club has disbursed
# (i.e., lent) over the years to the different risk
# markets (grade variable)

fig = filtered_loandata.groupby(['issue_d', 'loan_status_numeric'])['loan_amnt'].sum().unstack().plot(
    figsize=(14, 8), linewidth=2)

fig.set_title('Disbursed amount in time')
fig.set_ylabel('Disbursed Amount (US Dollars)')
filtered_loandata.loc[filtered_loandata.loan_amnt<10000]
plt.subplots(figsize=(12,6))
sns.violinplot(x="grade", y="loan_amnt", hue='loan_status_numeric', split=True, inner="quart", data=filtered_loandata, order="ABCDEFG")
sns.despine(left=True)
plt.subplots(figsize=(12,8))
sns.countplot("grade",hue='loan_status_numeric',data=filtered_loandata)
plt.subplots(figsize=(12,8))
sns.countplot("emp_length",hue='loan_status_numeric',data=filtered_loandata)
plt.subplots(figsize=(12,8))
sns.countplot("purpose",hue='loan_status_numeric',data=filtered_loandata)
loan_status_purpose=pd.crosstab(filtered_loandata['purpose'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_purpose.div(loan_status_purpose.sum(1).astype(float), axis=0).plot(kind="bar",figsize=(12,8))
plt.xlabel('purpose')
P = plt.ylabel('Percentage')
fig = filtered_loandata.groupby(['grade', 'term'])['loan_status_numeric'].mean().unstack().plot.bar()
fig.set_ylabel('Percentage of default')
loan_status_ownership=pd.crosstab(filtered_loandata['home_ownership'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_ownership
loan_status_ownership.div(loan_status_ownership.sum(1).astype(float), axis=0).plot(kind="bar")
plt.xlabel('home_ownership')
P = plt.ylabel('Percentage')
loan_status_ownership=pd.crosstab(filtered_loandata['grade'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_ownership
loan_status_ownership.div(loan_status_ownership.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True)
plt.xlabel('grade')
P = plt.ylabel('Percentage')
fig = filtered_loandata.groupby(['grade'])['loan_status_numeric'].mean().sort_values().plot.bar()
fig.set_ylabel('Percentage of bad debt')
loan_status_ownership=pd.crosstab(filtered_loandata['sub_grade'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_ownership
loan_status_ownership.div(loan_status_ownership.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True)
plt.xlabel('sub_grade')
P = plt.ylabel('Percentage')
plt.subplots(figsize=(12,8))
sns.countplot("verification_status",hue='loan_status_numeric',data=filtered_loandata)
print(filtered_loandata.duplicated().sum())
filtered_loandata.loan_amnt.describe()
bins = np.linspace(-1, 40000, 100)
plt.subplots(figsize=(12,8))
plt.hist(filtered_loandata.loan_amnt, bins, alpha=0.5, label='x')
plt.hist(filtered_loandata.loan_amnt[filtered_loandata.loan_status_numeric == 1], bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
filtered_loandata.loc[filtered_loandata.total_rec_late_fee>0]
fees = filtered_loandata.loc[filtered_loandata.total_rec_late_fee>0]
type(fees)
bins = np.linspace(-1, 200, 20)
plt.subplots(figsize=(12,8))
plt.hist(fees.total_rec_late_fee, bins, alpha=0.5, label='x')
plt.hist(fees.total_rec_late_fee[fees.loan_status_numeric == 1], bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
filtered_loandata.inq_last_6mths.value_counts()
loan_status_inquiries=pd.crosstab(filtered_loandata['inq_last_6mths'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_inquiries.div(loan_status_inquiries.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True, figsize=(18,8))
plt.xlabel('loan_status_inquiries')
P = plt.ylabel('Percentage')
filtered_loandata.delinq_2yrs.value_counts()
loan_status_inquiries=pd.crosstab(filtered_loandata['delinq_2yrs'],filtered_loandata['loan_status_numeric'], dropna = True)
loan_status_inquiries.div(loan_status_inquiries.sum(1).astype(float), axis=0).plot(kind="bar", stacked = True, figsize=(18,8))
plt.xlabel('loan_status_inquiries')
P = plt.ylabel('Percentage')
bins = np.linspace(-1, 100, 10)
plt.hist(filtered_loandata.dti, bins)
#plt.xscale('log')
plt.show()
bins = np.linspace(-1, 30, 30)
plt.subplots(figsize=(12,8))
plt.hist(filtered_loandata.dti, bins, alpha=0.5, label='x')
plt.hist(filtered_loandata.dti[filtered_loandata.loan_status_numeric == 1], bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
filtered_loandata.loc[filtered_loandata['annual_inc'] > 200000]
bins = np.linspace(-1, 50000, 20)
plt.subplots(figsize=(12,8))
plt.hist(filtered_loandata.revol_bal, bins, alpha=0.5, label='x')
plt.hist(filtered_loandata.revol_bal[filtered_loandata.loan_status_numeric == 1], bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
bins = np.linspace(-1, 300000, 30)
plt.subplots(figsize=(12,8))
plt.hist(filtered_loandata.annual_inc, bins, alpha=0.5, label='x')
plt.hist(filtered_loandata.annual_inc[filtered_loandata.loan_status_numeric == 1], bins, alpha=0.5, label='y')
plt.legend(loc='upper right')
plt.show()
# create new dataset for exploratory analysis

def categorize(l):
    uniques = sorted(list(set(l)))
    return [uniques.index(x) + 1 for x in l]

loan_exploratory = pd.DataFrame()
loan_exploratory['annual_inc'] = filtered_loandata['annual_inc']
loan_exploratory['delinq_2yrs'] = filtered_loandata['delinq_2yrs']
loan_exploratory['dti'] = filtered_loandata['dti']
loan_exploratory['grade'] = categorize(filtered_loandata['grade'])
loan_exploratory['home_ownership'] = categorize(filtered_loandata['home_ownership'])
loan_exploratory['installment'] = filtered_loandata['installment']
loan_exploratory['int_rate'] = filtered_loandata['int_rate']
loan_exploratory['loan_amnt'] = filtered_loandata['loan_amnt']
loan_exploratory['loan_status_numeric'] = filtered_loandata['loan_status_numeric']
loan_exploratory['verification_status'] = categorize(filtered_loandata['verification_status'])
loan_exploratory['sub_grade'] = categorize(filtered_loandata['sub_grade'])
loan_exploratory['term'] = categorize(filtered_loandata['term'])

loan_exploratory
plt.subplots(figsize=(20,12))
sns.set_context("paper", font_scale=2)
sns.heatmap(loan_exploratory.corr(), vmax=.8, square=True, annot=True, fmt='.1f')