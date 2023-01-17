import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

train=pd.read_csv('../input/bank-loan2/madfhantr.csv')

test=pd.read_csv('../input/bank-loan2/madhante.csv')

train_original=train.copy()

test_original=test.copy()
train.columns
test.columns
train.shape,test.shape
train.info()
train['Loan_Status'].value_counts()
train['Loan_Status'].value_counts(normalize=True)
plt.style.use('seaborn-whitegrid')

train['Loan_Status'].value_counts().plot.bar(width=0.2)

plt.xlabel("Value")

plt.title("Loan Staus")
plt.figure(figsize=(20,10))



plt.subplot(221)

train['Gender'].value_counts().plot.bar(title='Gender',width=0.2)

plt.subplot(222)

train['Married'].value_counts().plot.bar(title='Married',width=0.2)

plt.subplot(223)

train['Self_Employed'].value_counts().plot.bar(title='Self_Employed',width=0.2)

plt.subplot(224)

train['Credit_History'].value_counts().plot.bar(title='Credit_History',width=0.2)
plt.figure(figsize=(20,4))

plt.subplot(131)

train['Dependents'].value_counts(normalize=True).plot.bar(title= 'Dependents')

plt.subplot(132)

train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')

plt.subplot(133)

train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.distplot(train['ApplicantIncome'])

plt.subplot(122)

train['ApplicantIncome'].plot.box()
train.boxplot(column='ApplicantIncome', by = 'Education')

plt.suptitle("")
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.distplot(train['CoapplicantIncome'])

plt.subplot(122)

train['CoapplicantIncome'].plot.box()
plt.figure(figsize=(20,5))

plt.subplot(121)

sns.distplot(train['LoanAmount'])

plt.subplot(122)

train['LoanAmount'].plot.box()
gender=pd.crosstab(train['Gender'],train['Loan_Status'])

gender.div(gender.sum(1).astype(float),axis=0).plot.bar(stacked=True)
credit_his=pd.crosstab(train['Credit_History'],train['Loan_Status'])

credit_his.div(credit_his.sum(1).astype(float),axis=0).plot.bar(stacked=True)
prob_area=pd.crosstab(train['Property_Area'],train['Loan_Status'])

prob_area.div(prob_area.sum(1).astype('float'),axis=0).plot.bar(stacked=True)
train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()
bins=[0,2500,4000,6000,81000]

group=['Low','Average','High', 'Very high']

train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)

income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])

income_bin.div(income_bin.sum(1),axis=0).plot(kind="bar", stacked=True)
bins=[0,1000,3000,42000] 

group=['Low','Average','High'] 

train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)

Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])

Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype('float'),axis=0).plot.bar(stacked=True)
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']

bins=[0,2500,4000,6000,81000] 

group=['Low','Average','High', 'Very high'] 

train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)

Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 

Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 

plt.xlabel('Total_Income') 

P = plt.ylabel('Percentage')
bins=[0,100,200,700]

group=['Low','Average','High']

train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)

LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])

LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('LoanAmount') 

P = plt.ylabel('Percentage')
heat_map=train.corr()

sns.heatmap(heat_map, square=True, cmap="YlGnBu",annot=True)
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)

train['Married'].fillna(train['Married'].mode()[0], inplace=True) 

train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 

train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 

train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)

train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 

test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 

test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 

test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 

test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 

test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['LoanAmount_log'] = np.log(train['LoanAmount']) 

train['LoanAmount_log'].hist(bins=20) 

test['LoanAmount_log'] = np.log(test['LoanAmount'])