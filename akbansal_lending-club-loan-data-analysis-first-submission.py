import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date, datetime

import warnings
warnings.filterwarnings(action="ignore")
import os
print(os.listdir("../input"))

loanData= pd.read_csv('../input/LoanStats.csv',header=1,error_bad_lines=False,skipfooter=2,engine='python')
loanData.isnull().all()
loanData.dropna(axis=1,how='all',inplace=True)  # drop all null columns
pd.set_option('display.max_columns', 65)
loanData.head()
loanData.drop(['emp_title','desc','title','application_type'],axis=1,inplace=True) #carries no useful info which can affect loan terms
loanData['id']=np.arange(1,42537)


print(loanData.policy_code.unique())  # drop it since it contains only 1 value so doesn't make any impact
print(loanData.loan_status.unique())
print(loanData.home_ownership.unique())
print(loanData.verification_status.unique())
print(loanData.pymnt_plan.unique())  ## drop it since it contains only 1 value i.e 'n' , so doesn't make any impact
print(loanData.disbursement_method.unique())   # drop it since it contains only 1 value i.e 'Cash' , so doesn't make any impact
print(loanData.hardship_flag.unique())         # drop it since it contains only 1 value i.e 'N' , so doesn't make any impact
print(loanData.tax_liens.unique())
print(loanData.pub_rec_bankruptcies.unique())
print(loanData.initial_list_status.unique())   #  drop it since it contains only 1 value i.e 'f' , so doesn't make any impact



loanData.drop(['policy_code','pymnt_plan','disbursement_method','hardship_flag','initial_list_status'],axis=1,inplace=True) # since contain only 1 type of value so doesn't make any impact

loanData['int_rate']=loanData.int_rate.str.extract('(\d+.\d+)')

loanData['int_rate']=loanData.int_rate.astype('float64')
loanData.int_rate.dtype
loanData.loc[loanData.loan_status.str.contains('Fully Paid',na=False),'loan_status']='Paid'   
loanData.loc[loanData.loan_status.str.contains('Charged Off',na=False),'loan_status']='ChargedOff'  
loanData.loan_status.unique()

loanData['loan_status'] =loanData.loan_status.astype('category')
loanData.loan_status.unique()
loanData.loc[loanData.verification_status.str.contains('Not Verified',na=False),'verification_status']='NotVerify'  
loanData.loc[loanData.verification_status.str.contains('Verified',na=False),'verification_status']= 'Verify'   

loanData['verification_status'] =loanData.verification_status.astype('category')
loanData.verification_status.unique()
loanData['term'] =loanData.term.str.extract('(\d+)')
loanData['emp_length'] =loanData.emp_length.str.extract('(\d+)')
loanData['revol_util'] =loanData.revol_util.str.extract('(\d+.\d+)')
loanData['sub_grade']=loanData.sub_grade.str.extract('(\d+)')

loanData.term.dropna(inplace=True)
loanData['term']=loanData.term.astype('int')
#loanData['grade']=loanData.grade.astype('category')
loanData['home_ownership']=loanData.home_ownership.astype('category')
loanData['revol_util']=loanData.revol_util.astype('float')
#loanData['sub_grade']=loanData.sub_grade.astype('int')

loanData.emp_length.fillna(0,inplace=True)
loanData['emp_length']=loanData.emp_length.astype('int')
loanData.emp_length.unique()
# handling date columns

loanData['issue_d'] = pd.to_datetime(loanData['issue_d'])
loanData['earliest_cr_line'] = pd.to_datetime(loanData['earliest_cr_line'])
loanData['last_pymnt_d'] = pd.to_datetime(loanData['last_pymnt_d'])
loanData['last_credit_pull_d'] = pd.to_datetime(loanData['last_credit_pull_d'])
loanData['settlement_date'] = pd.to_datetime(loanData['settlement_date'])
loanData['debt_settlement_flag_date'] = pd.to_datetime(loanData['debt_settlement_flag_date'])
loanData['next_pymnt_d'] = pd.to_datetime(loanData['next_pymnt_d'])

             
loanData.loc[loanData.settlement_amount.notnull(),['loan_amnt','issue_d','loan_status','settlement_status','settlement_amount','settlement_percentage','settlement_term','settlement_date','debt_settlement_flag_date ']]

loanData.drop(['debt_settlement_flag_date','settlement_term','settlement_status','settlement_date','settlement_percentage','settlement_amount'],axis=1,inplace=True) # since doesn't have enough values 

loanData.tax_liens.value_counts()

loanData.drop('tax_liens',axis=1,inplace=True)
df=loanData.loc[:30000,['loan_amnt','funded_amnt','funded_amnt_inv']]
sns.pairplot(vars=['loan_amnt','funded_amnt','funded_amnt_inv'],data=df)
df=loanData.groupby('purpose').id.count().reset_index()
df.rename(columns={'id':'no_of_loans'},inplace=True)
plt.figure(figsize=(10,10))
plt.pie(df.no_of_loans,labels=df.purpose,autopct='%.2f%%');
df.plot.bar(x='purpose',y='no_of_loans',figsize=(10,6));

df=loanData.groupby('addr_state').id.count().sort_values().reset_index()
df.plot.bar('addr_state','id',figsize=(15,6))
plt.title('loan distribution by state')
df.head()
df=loanData.groupby(loanData.issue_d.dt.year).int_rate.mean()
df.plot(kind='line',figsize=(10,6))
plt.title('Change in Avg. Interest Rate by year');



df=loanData.groupby(loanData.issue_d.dt.year).id.count()
df.plot(kind='line',figsize=(10,6))
plt.title('Total loans taken per year');
plt.ylabel('No of loans issued')
df.head()

df=loanData.groupby(loanData.issue_d.dt.month).id.count().sort_values()
df.plot(kind='line',figsize=(8,6));
plt.title('Total No of loans taken by month ')
plt.xlabel('Month');
plt.ylabel('No of loans issued');

# maximum loans are taken during end of the year
df=loanData.groupby('grade').loan_status.value_counts().unstack()
df.plot.bar()
plt.title('No of loans paid and charged_off according to grade');
loanData.pivot_table(index='grade',columns='sub_grade',values='int_rate').plot.bar(figsize=(10,6))
plt.ylabel('Interest rate');


plt.figure(figsize=(10,6))
sns.boxplot(data=loanData, x='grade',y='int_rate',hue='loan_status')
plt.title('Interest Rate IQR(range) with grade');
df=loanData.groupby('home_ownership').loan_status.value_counts().unstack()
df.plot.bar(figsize=(10,6))
plt.title('No of loans paid and charged off according to home ownership')
df
sns.countplot(data=loanData,x='verification_status')
plt.title('Source verification of loans');

plt.figure(figsize=(10,6))

sns.distplot(loanData.loan_amnt.fillna(0))

