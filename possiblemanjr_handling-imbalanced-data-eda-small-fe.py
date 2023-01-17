import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec # Alignments 



import seaborn as sns # theme & dataset

print(f"Matplotlib Version : {mpl.__version__}")

print(f"Seaborn Version : {sns.__version__}")





plt.style.use('ggplot')

%matplotlib inline
df = pd.read_csv('/kaggle/input/loan-data-for-dummy-bank/loan_final313.csv')
df.head()
plt.rcParams['figure.dpi'] = 200
df.columns
df.info()
df.isnull().sum()
## Check how imbalanced TARGET is

df['loan_condition'].value_counts()
# to recognize what is designated as _cat

df3 = df.loc[df['grade'] == 'G']

df3['grade_cat'].head()
f,ax=plt.subplots(1,2,figsize=(18,8))

df['loan_condition'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)

ax[0].set_title('loan_condition')

ax[0].set_ylabel('')

sns.countplot('loan_condition',data=df,ax=ax[1])

ax[1].set_title('loan_condition')

plt.show()
df.groupby(['home_ownership','loan_condition'])['loan_condition'].count()
f,ax=plt.subplots(1,2,figsize=(18,8))

df[['home_ownership','loan_condition_cat']].groupby(['home_ownership']).mean().plot.bar(ax=ax[0])

ax[0].set_title('loan_condition vs home_ownership')

sns.countplot('home_ownership',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('home_ownership:Bad Loan vs Good loan')

plt.show()
df.groupby(['term','loan_condition'])['loan_condition'].count()
pd.crosstab(df.term,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df[['term','loan_condition_cat']].groupby(['term']).mean().plot.bar(ax=ax[0])

ax[0].set_title('loan_condition vs term')

sns.countplot('term',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('term:Bad Loan vs Good loan')

plt.show()
pd.crosstab(df.application_type,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab(df.purpose,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df[['purpose','loan_condition_cat']].groupby(['purpose']).mean().plot.bar(ax=ax[0])

ax[0].set_title('loan_condition vs purpose')

sns.countplot('purpose_cat',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('purpose:Bad Loan vs Good Loan')

plt.show()



####purpose_cat = 6 refers to debt_consolidation
pd.crosstab(df.region,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df[['region','loan_condition_cat']].groupby(['region']).mean().plot.bar(ax=ax[0])

ax[0].set_title('loan_condition vs region')

sns.countplot('region',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('Region : ')

plt.show()
pd.crosstab(df.emp_length_int,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df['emp_length_int'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Loans By emp_length_int')

ax[0].set_ylabel('Count')

sns.countplot('emp_length_int',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('emp_length_int: Bad Loan vs Good Loan')

plt.show()
pd.crosstab(df.grade,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df['grade'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Loans By Grade')

ax[0].set_ylabel('Count')

sns.countplot('grade',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('grade: Bad Loan vs Good Loan')

plt.show()
pd.crosstab(df.interest_payments,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df['interest_payments'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Loans By Interest_payments')

ax[0].set_ylabel('Count')

sns.countplot('interest_payments',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('interest_payments: Bad Loan vs Good Loan')

plt.show()
pd.crosstab(df.income_category,df.loan_condition,margins=True).style.background_gradient(cmap='summer_r')
f,ax=plt.subplots(1,2,figsize=(18,8))

df['income_category'].value_counts().plot.bar(color=['#CD7F32','#FFDF00','#D3D3D3'],ax=ax[0])

ax[0].set_title('Number Of Loans By Income_category')

ax[0].set_ylabel('Count')

sns.countplot('income_category',hue='loan_condition_cat',data=df,ax=ax[1])

ax[1].set_title('Income_category: Bad Loan vs Good Loan')

plt.show()
print('Highest Loan Amount was:',df['loan_amount'].max(),'$')

print('Lowest Loan Amount was:',df['loan_amount'].min(),'$')

print('Average Loan Amount was:',df['loan_amount'].mean(),'$')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("income_category","loan_amount", hue="loan_condition_cat", data=df,split=True,ax=ax[0])

ax[0].set_title('income_category and loan_amount vs Loan_condition')

ax[0].set_yticks(range(0,40000,5000))

sns.violinplot("term","loan_amount", hue="loan_condition_cat", data=df,split=True,ax=ax[1])

ax[1].set_title('Term and loan_amount vs Loan_condition')

ax[1].set_yticks(range(0,40000,5000))

plt.show()
print('Highest installment was:',df['installment'].max(),'$')

print('Lowest installment was:',df['installment'].min(),'$')

print('Average installment was:',df['installment'].mean(),'$')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("income_category","installment", hue="loan_condition_cat", data=df,split=True,ax=ax[0])

ax[0].set_title('income_category and installment vs Loan_condition')

ax[0].set_yticks(range(0,1500,300))

sns.violinplot("term","installment", hue="loan_condition_cat", data=df,split=True,ax=ax[1])

ax[1].set_title('Term and installment vs Loan_condition')

ax[1].set_yticks(range(0,1500,300))

plt.show()
print('Highest interest_rate was:',df['interest_rate'].max(),'%')

print('Lowest interest_rate was:',df['interest_rate'].min(),'%')

print('Average interest_rate was:',df['interest_rate'].mean(),'%')
f,ax=plt.subplots(1,2,figsize=(18,8))

sns.violinplot("income_category","interest_rate", hue="loan_condition_cat", data=df,split=True,ax=ax[0])

ax[0].set_title('income_category and interest_rate vs Loan_condition')

ax[0].set_yticks(range(0,30,3))

sns.violinplot("term","interest_rate", hue="loan_condition_cat", data=df,split=True,ax=ax[1])

ax[1].set_title('Term and interest_rate vs Loan_condition')

ax[1].set_yticks(range(0,30,3))

plt.show()
df_labeled = df.drop(['id','final_d', 'year','loan_condition', 'issue_d', 'home_ownership', 'income_category', 

              'term', 'application_type', 'purpose', 'interest_payments', 'loan_condition', 'grade', 'region'],axis=1 )
sns.heatmap(df_labeled.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df.corr()-->correlation matrix

fig=plt.gcf()

fig.set_size_inches(20, 16)

plt.show()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df['region'] = le.fit_transform(df['region'])
df_for_use = df.drop(['id','final_d', 'year','loan_condition', 'issue_d', 'home_ownership', 'income_category', 

              'term', 'application_type', 'purpose', 'interest_payments', 'loan_condition', 'grade'],axis=1 )
df_for_use.head()
#### Export





df_for_use.to_pickle('df_for_use.pkl')
df_fe = df.drop(['id','final_d', 'year','loan_condition', 'issue_d', 'home_ownership', 'income_category', 'term', 'application_type', 'purpose', 'interest_payments', 'loan_condition', 'grade', 'region'],axis=1 )
df_fe.columns
# df_fe.drop(['total_rec_prncp', 'installment', 'interest_rate' ] , axis = 1, inplace = True)
df_fe['LoanAmntOverIncome'] = df_fe['loan_amount'] / df_fe['annual_inc']

df_fe['installmentOverLoanAmnt'] = df_fe['installment'] / df_fe['loan_amount']

df_fe['totalPymntOverIncome'] = df_fe['total_pymnt'] / df_fe['annual_inc']

df_fe['totalRecPrncpOverIncome'] = df_fe['total_rec_prncp'] / df_fe['annual_inc']



df.dtypes
df_fe.astype({'LoanAmntOverIncome': 'float32'}).dtypes
df_fe['totalRecPrncpOverIncome']
nanCounter = np.isnan(df_fe.loc[:,df_fe.columns]).sum()
nanCounter
df_fe.head()
#### Export





df_fe.to_pickle('df_fe.pkl')