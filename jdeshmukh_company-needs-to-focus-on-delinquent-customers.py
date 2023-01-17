
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
%matplotlib inline 

import warnings
warnings.filterwarnings("ignore")
# Reading data

train_df=pd.read_csv("../input/av-janata-hack-payment-default-prediction/train_20D8GL3.csv") 
test_df=pd.read_csv("../input/av-janata-hack-payment-default-prediction/test_O6kKpvt.csv")

original_train_df=train_df.copy() 
original_test_df=test_df.copy()
train_df.columns
test_df.columns
train_df.dtypes
test_df.dtypes
train_df.shape, test_df.shape
train_df.head()
test_df.head()
train_df.rename(columns = {'PAY_0':'PAY_1'}, inplace = True)
test_df.rename(columns = {'PAY_0':'PAY_1'}, inplace = True)
train_df['MIN_AMT6']=train_df['BILL_AMT6']*0.1
test_df['MIN_AMT6']=test_df['BILL_AMT6']*0.1
train_df['PENDING_AMT6']=train_df['BILL_AMT6'] - train_df['PAY_AMT6']
test_df['PENDING_AMT6']=test_df['BILL_AMT6'] - test_df['PAY_AMT6']
train_df['MIN_AMT5']=train_df['PENDING_AMT6']*0.1
test_df['MIN_AMT5']=test_df['PENDING_AMT6']*0.1
train_df['DELINQ_5'] = np.where((train_df['PAY_AMT5']>0) & (train_df['PAY_AMT5']<train_df['MIN_AMT6']),1,0)
test_df['DELINQ_5'] = np.where((test_df['PAY_AMT5']>0) & (test_df['PAY_AMT5']<test_df['MIN_AMT6']),1,0)
train_df['NO_PMNT5']=np.where(train_df['PAY_AMT5'] == 0,1,0)
test_df['NO_PMNT5']=np.where(test_df['PAY_AMT5'] == 0,1,0)
train_df['PENDING_AMT5'] = (train_df['BILL_AMT5']+train_df['BILL_AMT6']) - (train_df['PAY_AMT5']+train_df['PAY_AMT6'])
test_df['PENDING_AMT5'] = (test_df['BILL_AMT5']+test_df['BILL_AMT6']) - (test_df['PAY_AMT5']+test_df['PAY_AMT6'])
train_df['MIN_AMT4']=train_df['PENDING_AMT5']*0.1
test_df['MIN_AMT4']=test_df['PENDING_AMT5']*0.1
train_df['DELINQ_4'] = np.where((train_df['PAY_AMT4']>0) & (train_df['PAY_AMT4']<train_df['MIN_AMT5']),1,0) + train_df['DELINQ_5']
test_df['DELINQ_4'] = np.where((test_df['PAY_AMT4']>0) & (test_df['PAY_AMT4']<test_df['MIN_AMT5']),1,0) + test_df['DELINQ_5']
train_df['NO_PMNT4']=np.where(train_df['PAY_AMT4'] == 0,1,0) + train_df['NO_PMNT5']
test_df['NO_PMNT4']=np.where(test_df['PAY_AMT4'] == 0,1,0) + test_df['NO_PMNT5']
train_df['PENDING_AMT4'] = (train_df['BILL_AMT4']+train_df['BILL_AMT5']+train_df['BILL_AMT6']) - (train_df['PAY_AMT4']+train_df['PAY_AMT5']+train_df['PAY_AMT6'])
test_df['PENDING_AMT4'] = (test_df['BILL_AMT4']+test_df['BILL_AMT5']+test_df['BILL_AMT6']) - (test_df['PAY_AMT4']+test_df['PAY_AMT5']+test_df['PAY_AMT6'])
train_df['MIN_AMT3']=train_df['PENDING_AMT4']*0.1
test_df['MIN_AMT3']=test_df['PENDING_AMT4']*0.1
train_df['DELINQ_3'] = np.where((train_df['PAY_AMT3']>0) & (train_df['PAY_AMT3']<train_df['MIN_AMT4']),1,0) + train_df['DELINQ_4']
test_df['DELINQ_3'] = np.where((test_df['PAY_AMT3']>0) & (test_df['PAY_AMT3']<test_df['MIN_AMT4']),1,0) + test_df['DELINQ_4']
train_df['NO_PMNT3']=np.where(train_df['PAY_AMT3'] == 0,1,0) + train_df['NO_PMNT4']
test_df['NO_PMNT3']=np.where(test_df['PAY_AMT3'] == 0,1,0) + test_df['NO_PMNT4']
train_df['PENDING_AMT3'] = (train_df['BILL_AMT3']+train_df['BILL_AMT4']+train_df['BILL_AMT5']+train_df['BILL_AMT6']) - (train_df['PAY_AMT3']+train_df['PAY_AMT4']+train_df['PAY_AMT5']+train_df['PAY_AMT6'])
test_df['PENDING_AMT3'] = (test_df['BILL_AMT3']+test_df['BILL_AMT4']+test_df['BILL_AMT5']+test_df['BILL_AMT6']) - (test_df['PAY_AMT3']+test_df['PAY_AMT4']+test_df['PAY_AMT5']+test_df['PAY_AMT6'])
train_df['MIN_AMT2']=train_df['PENDING_AMT3']*0.1
test_df['MIN_AMT2']=test_df['PENDING_AMT3']*0.1
train_df['DELINQ_2'] = np.where((train_df['PAY_AMT2']>0) & (train_df['PAY_AMT2']<train_df['MIN_AMT3']),1,0) + train_df['DELINQ_3']
test_df['DELINQ_2'] = np.where((test_df['PAY_AMT2']>0) & (test_df['PAY_AMT2']<test_df['MIN_AMT3']),1,0) + test_df['DELINQ_3']
train_df['NO_PMNT2']=np.where(train_df['PAY_AMT2'] == 0,1,0) + train_df['NO_PMNT3']
test_df['NO_PMNT2']=np.where(test_df['PAY_AMT2'] == 0,1,0) + test_df['NO_PMNT3']
train_df['PENDING_AMT2'] = (train_df['BILL_AMT2']+train_df['BILL_AMT3']+train_df['BILL_AMT4']+train_df['BILL_AMT5']+train_df['BILL_AMT6']) - (train_df['PAY_AMT2']+train_df['PAY_AMT3']+train_df['PAY_AMT4']+train_df['PAY_AMT5']+train_df['PAY_AMT6'])
test_df['PENDING_AMT2'] = (test_df['BILL_AMT2']+test_df['BILL_AMT3']+test_df['BILL_AMT4']+test_df['BILL_AMT5']+test_df['BILL_AMT6']) - (test_df['PAY_AMT2']+test_df['PAY_AMT3']+test_df['PAY_AMT4']+test_df['PAY_AMT5']+test_df['PAY_AMT6'])
train_df['MIN_AMT1']=train_df['PENDING_AMT2']*0.1
test_df['MIN_AMT1']=test_df['PENDING_AMT2']*0.1
train_df['DELINQ_1'] = np.where((train_df['PAY_AMT1']>0) & (train_df['PAY_AMT1']<train_df['MIN_AMT2']),1,0) + train_df['DELINQ_2']
test_df['DELINQ_1'] = np.where((test_df['PAY_AMT1']>0) & (test_df['PAY_AMT1']<test_df['MIN_AMT2']),1,0) + test_df['DELINQ_2']
train_df['NO_PMNT1']=np.where(train_df['PAY_AMT1'] == 0,1,0) + train_df['NO_PMNT2']
test_df['NO_PMNT1']=np.where(test_df['PAY_AMT1'] == 0,1,0) + test_df['NO_PMNT2']
train_df['PENDING_AMT1'] = (train_df['BILL_AMT1']+train_df['BILL_AMT2']+train_df['BILL_AMT3']+train_df['BILL_AMT4']+train_df['BILL_AMT5']+train_df['BILL_AMT6']) - (train_df['PAY_AMT1']+train_df['PAY_AMT2']+train_df['PAY_AMT3']+train_df['PAY_AMT4']+train_df['PAY_AMT5']+train_df['PAY_AMT6'])
test_df['PENDING_AMT1'] = (test_df['BILL_AMT1']+test_df['BILL_AMT2']+test_df['BILL_AMT3']+test_df['BILL_AMT4']+test_df['BILL_AMT5']+test_df['BILL_AMT6']) - (test_df['PAY_AMT1']+test_df['PAY_AMT2']+test_df['PAY_AMT3']+test_df['PAY_AMT4']+test_df['PAY_AMT5']+test_df['PAY_AMT6'])
train_df['AVG_6MTH_BAL'] = train_df['PENDING_AMT1']/6
test_df['AVG_6MTH_BAL'] = test_df['PENDING_AMT1']/6
train_df['CREDIT_UTIL_RATIO'] = train_df['AVG_6MTH_BAL']/train_df['LIMIT_BAL']
test_df['CREDIT_UTIL_RATIO'] = test_df['AVG_6MTH_BAL']/test_df['LIMIT_BAL']
bins=[0,20,30,40,50,60,70,80] 
group=['VERY_YOUNG','YOUNG','MIDDLE','SENIOR','VERY_SENIOR','RETIRED','ELDERLY'] 

train_df['AGE_BIN']=pd.cut(train_df['AGE'],bins,labels=group)
test_df['AGE_BIN']=pd.cut(test_df['AGE'],bins,labels=group)
original_columns = train_df.columns
# delete un-necessary columns

columns_to_delete = ['ID','AGE']
final_columns = list(set(original_columns)-set(columns_to_delete))
final_columns
final_train_df = train_df[final_columns]
original_columns = test_df.columns
final_columns = list(set(original_columns)-set(columns_to_delete))
final_columns
final_test_df = test_df[final_columns]
final_train_df.shape, final_test_df.shape
# Let us look at the datatypes of the training data columns.

final_train_df.dtypes
# Let us look at the datatypes of the test data columns.

final_test_df.dtypes
total = final_train_df.isnull().sum().sort_values(ascending = False)
percent = (final_train_df.isnull().sum()/final_train_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()
final_train_df.describe()
final_train_df['default_payment_next_month'].value_counts()
sns.countplot(final_train_df['default_payment_next_month'])
plt.show()
final_train_df['default_payment_next_month'].value_counts(normalize=True)
final_train_df['default_payment_next_month'].value_counts(normalize=True).plot.bar()
final_train_df['SEX'].value_counts(normalize=True).plot.bar(figsize=(10,5), title= 'SEX') 
plt.show()

plt.figure(1) 
plt.subplot(221) 
final_train_df['MARRIAGE'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'MARRIAGE') 
plt.subplot(222) 
final_train_df['EDUCATION'].value_counts(normalize=True).plot.bar(title= 'EDUCATION') 
plt.show()
plt.figure(1)
plt.subplot(221) 
final_train_df['AGE_BIN'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'AGE') 
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['LIMIT_BAL']); 
plt.subplot(122) 
final_train_df['LIMIT_BAL'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.LIMIT_BAL, bins = bins, label = 'Total', alpha=0.5)

# plt.hist(data.LIMIT_BAL[data['default.payment.next.month'] == 1], bins = bins, color='b',label = 'Default')

plt.xlabel('Credit Limit (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Fig.1 : Credit Limit ',fontweight="bold", size=12)
plt.legend();
plt.show()
len(final_train_df[final_train_df['LIMIT_BAL']>= 600000])
len(final_train_df[(final_train_df['LIMIT_BAL']>= 600000) & (final_train_df['default_payment_next_month']== 1)])
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT1']); 
plt.subplot(122) 
final_train_df['BILL_AMT1'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT1, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount Sep-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount Sep-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT2']); 
plt.subplot(122) 
final_train_df['BILL_AMT2'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT2, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount Aug-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount Aug-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT3']); 
plt.subplot(122) 
final_train_df['BILL_AMT3'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT3, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount July-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount July-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT4']); 
plt.subplot(122) 
final_train_df['BILL_AMT4'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT4, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount June-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount June-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT5']); 
plt.subplot(122) 
final_train_df['BILL_AMT5'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT5, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount May-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount May-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['BILL_AMT6']); 
plt.subplot(122) 
final_train_df['BILL_AMT6'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.BILL_AMT6, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Bill Amount April-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Bill Amount April-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
final_train_df[final_train_df['BILL_AMT1'] <0][['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']]
final_train_df[final_train_df['BILL_AMT2'] <0][['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']]
for billamt in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']:
    print("\nNo. of Customers with ", billamt, " >= 200000 : ", len(final_train_df[final_train_df[billamt] >=200000]))
for billamt in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']:
    print("\nNo. of Customers with ", billamt, " >= 400000 : ", len(final_train_df[final_train_df[billamt] >=400000]))
for billamt in ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']:
    print("\nNo. of Customers with ", billamt, " >= 600000 : ", len(final_train_df[final_train_df[billamt] >=600000]))
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT1']); 
plt.subplot(122) 
final_train_df['PAY_AMT1'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT1, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount Sept-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Payment Amount Sept-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT2']); 
plt.subplot(122) 
final_train_df['PAY_AMT2'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT2, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount Aug-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Payment Amount Aug-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT3']); 
plt.subplot(122) 
final_train_df['PAY_AMT3'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT3, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount July-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Payment Amount July-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT4']); 
plt.subplot(122) 
final_train_df['PAY_AMT4'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT4, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount June-2005 ');plt.ylabel('Number of Accounts')
plt.title('Payment Amount June-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT5']); 
plt.subplot(122) 
final_train_df['PAY_AMT5'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT5, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount May-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Payment Amount May-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PAY_AMT6']); 
plt.subplot(122) 
final_train_df['PAY_AMT6'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PAY_AMT6, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Payment Amount April-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Payment Amount April-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
for pmnt in ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',]:
    print("\nNo. of Customers with ", pmnt, " >= 100000 : ", len(final_train_df[final_train_df[pmnt]>=100000]))
# Let's start by visualizing the distribution of 'DELINQ_1' in the dataset.  

fig, ax = plt.subplots()

x = final_train_df.DELINQ_1.unique()

# Counting total delinquencies in the dataset

y = final_train_df.DELINQ_1.value_counts()

# Plotting the bar graph

ax.bar(x, y)
ax.set_xlabel('Total Number of Delinquencies in last 6 months.')
ax.set_ylabel('No. of Customers')
plt.show()

# Let's start by visualizing the distribution of 'NO_PMNT1' in the dataset.  

fig, ax = plt.subplots()

x = final_train_df.NO_PMNT1.unique()

# Counting total delinquencies in the dataset

y = final_train_df.NO_PMNT1.value_counts()

# Plotting the bar graph

ax.bar(x, y)
ax.set_xlabel('Total Number of No Payments in last 6 months.')
ax.set_ylabel('No. of Customers')
plt.show()

plt.figure(1) 
plt.subplot(221) 
final_train_df['DELINQ_1'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Total Number of Delinquencies in last 6 months.') 
plt.subplot(222) 
final_test_df['NO_PMNT1'].value_counts(normalize=True).plot.bar(title= 'Total Number of No Payments in last 6 months.') 
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PENDING_AMT5']); 
plt.subplot(122) 
final_train_df['PENDING_AMT5'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PENDING_AMT5, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Pending Amount till May-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Pending Amount till May-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PENDING_AMT4']); 
plt.subplot(122) 
final_train_df['PENDING_AMT4'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PENDING_AMT4, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Pending Amount till June-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Pending Amount till June-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PENDING_AMT3']); 
plt.subplot(122) 
final_train_df['PENDING_AMT3'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PENDING_AMT3, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Pending Amount till July-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Pending Amount till July-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PENDING_AMT2']); 
plt.subplot(122) 
final_train_df['PENDING_AMT2'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PENDING_AMT2, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Pending Amount till August-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Pending Amount till August-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['PENDING_AMT1']); 
plt.subplot(122) 
final_train_df['PENDING_AMT1'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.PENDING_AMT1, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Pending Amount till Sept-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Pending Amount till Sept-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
plt.figure(1) 
plt.subplot(121) 
sns.distplot(final_train_df['AVG_6MTH_BAL']); 
plt.subplot(122) 
final_train_df['AVG_6MTH_BAL'].plot.box(figsize=(16,5)) 
plt.show()
bins = 20
plt.hist(final_train_df.AVG_6MTH_BAL, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Avg. 6 Month Balance in Sept-2005 (NT dollar)');plt.ylabel('Number of Accounts')
plt.title('Avg. 6 Month Balance in Sept-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
print("\nTotal Credit given to all the customers (as on Sept-2005): ", final_train_df['LIMIT_BAL'].sum())
print("\nTotal Credit given to all Defaulting customers (as on Sept-2005): ", final_train_df.groupby('default_payment_next_month')['LIMIT_BAL'].sum()[1])
print("\nTotal Pending Amount from all the customers (as on Sept-2005): ", final_train_df['PENDING_AMT1'].sum())
print("\nTotal Pending Amount from all Defaulting customers (as on Sept-2005): ", final_train_df.groupby('default_payment_next_month')['PENDING_AMT1'].sum()[1])
delinquencies = list(final_train_df['DELINQ_1'].unique())
for num_delinq in delinquencies:
    
    print("\nTotal Credit Amount for : ", num_delinq, " delinquencies is : ", sum(final_train_df[final_train_df['DELINQ_1']==num_delinq]['LIMIT_BAL']))
    
    print("\nTotal Pending Amount for : ", num_delinq, " delinquencies is : ", sum(final_train_df[final_train_df['DELINQ_1']==num_delinq]['PENDING_AMT1']))
              
nopayments = list(final_train_df['NO_PMNT1'].unique())
for num_pmnt in nopayments:
    
    print("\nTotal Credit Amount for : ", num_pmnt, " no payments is : ", sum(final_train_df[final_train_df['NO_PMNT1']==num_pmnt]['LIMIT_BAL']))
    
    print("\nTotal Pending Amount for : ", num_pmnt, " no payments is : ", sum(final_train_df[final_train_df['NO_PMNT1']==num_pmnt]['PENDING_AMT1']))

bins = 20
plt.hist(final_train_df.CREDIT_UTIL_RATIO, bins = bins, label = 'Total', alpha=0.8)

plt.xlabel('Credit Utilization Ratios in Sept-2005');plt.ylabel('Number of Accounts')
plt.title('Credit Utilization Ratios in Sept-2005 ',fontweight="bold", size=12)
plt.legend();
plt.show()
len(final_train_df[final_train_df['CREDIT_UTIL_RATIO'] <= 0.3])
len(final_train_df[(final_train_df['CREDIT_UTIL_RATIO'] > 0.3) & (final_train_df['CREDIT_UTIL_RATIO'] <= 0.7)])
len(final_train_df[(final_train_df['CREDIT_UTIL_RATIO'] > 0.7) & (final_train_df['CREDIT_UTIL_RATIO'] <= 1)])
len(final_train_df[final_train_df['CREDIT_UTIL_RATIO'] > 1])
corr = final_train_df.corr()
corr['default_payment_next_month']
corr['default_payment_next_month']>= 0.1
# Making correlation coefficients pair plot of the selected features

selected_columns = ['CREDIT_UTIL_RATIO','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','NO_PMNT1','NO_PMNT2','NO_PMNT3','NO_PMNT4','NO_PMNT5','default_payment_next_month']
plt.figure(figsize=(20,20))
ax = plt.axes()
corr_selected = final_train_df[selected_columns].corr()
sns.heatmap(corr_selected, vmax=1,vmin=-1, square=True, annot=True, cmap='Spectral',linecolor="white", linewidths=0.01, ax=ax)
ax.set_title('Correlation Coefficient Pair Plot',fontweight="bold", size=20)
plt.show()
train_columns = ['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
target_column = ['default_payment_next_month']
X = final_train_df[train_columns]
X.shape
y = final_train_df['default_payment_next_month']
y.shape
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=2020)

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb

from sklearn import metrics 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score

from statistics import mean 

i=1 
kf = StratifiedKFold(n_splits=5,random_state=2020,shuffle=True) 

score_rf = []

X_train = np.array(X_train)
                     
for train_index,test_index in kf.split(X_train,y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train.iloc[train_index],y_train.iloc[test_index]
    model_rf = RandomForestClassifier(class_weight='balanced',random_state=2020)
    model_rf.fit(xtr, ytr)
    pred_test_rf = model_rf.predict(xvl)
    score_rf.append(accuracy_score(yvl,pred_test_rf))
    print('\nAccuracy_score : ',score_rf[i-1])
    i+=1 
    
print("\nThe mean validation accuracy of Random Forest model is : ", mean(score_rf))

# Output confusion matrix

pred_rf = model_rf.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, pred_rf))
print()
print("Classification Report")
print(classification_report(y_val, pred_rf))

# Visualize the ROC curve

pred_rf_prob=model_rf.predict_proba(xvl)[:,1]

fpr, tpr, _ = metrics.roc_curve(yvl,  pred_rf_prob)
auc = metrics.roc_auc_score(yvl, pred_rf_prob)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="Validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4) 
plt.show()

importances=pd.Series(model_rf.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(10,6))

i=1 

score_gb = []

for train_index,test_index in kf.split(X_train,y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train.iloc[train_index],y_train.iloc[test_index]
    
    model_gb = GradientBoostingClassifier(random_state=2020)
    
    model_gb.fit(xtr, ytr)
    pred_test_gb = model_gb.predict(xvl)
    score_gb.append(accuracy_score(yvl,pred_test_gb))
    print('\nAccuracy_score : ',score_gb[i-1])
    i+=1 
    
print("\nThe mean validation accuracy of the Gradient Boosting model is : ", mean(score_gb))

# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

pred_gb = model_gb.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, pred_gb))
print()
print("Classification Report")
print(classification_report(y_val, pred_gb))
# Visualize the ROC curve

pred_gb_prob=model_gb.predict_proba(xvl)[:,1]

fpr, tpr, _ = metrics.roc_curve(yvl,  pred_gb_prob)
auc = metrics.roc_auc_score(yvl, pred_gb_prob)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="Validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4) 
plt.show()

importances=pd.Series(model_gb.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))

i=1 

score_adb = []

for train_index,test_index in kf.split(X_train,y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train.iloc[train_index],y_train.iloc[test_index]
    model_adb = AdaBoostClassifier(random_state=2020)
    model_adb.fit(xtr, ytr)
    pred_test_adb = model_adb.predict(xvl)
    score_adb.append(accuracy_score(yvl,pred_test_adb))
    print('\nAccuracy_score : ',score_adb[i-1])
    i+=1 
    
print("\nThe mean validation accuracy of the Ada Boosting model is : ", mean(score_adb))

# Output confusion matrix and classification report of Ada Boosting algorithm on validation set

pred_adb = model_adb.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, pred_adb))
print()
print("Classification Report")
print(classification_report(y_val, pred_adb))
# Visualize the ROC curve

pred_adb_prob=model_adb.predict_proba(xvl)[:,1]

fpr, tpr, _ = metrics.roc_curve(yvl,  pred_adb_prob)
auc = metrics.roc_auc_score(yvl, pred_adb_prob)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="Validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4) 
plt.show()

importances=pd.Series(model_adb.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))



i=1 

score_xgb = []

for train_index,test_index in kf.split(X_train,y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train.iloc[train_index],y_train.iloc[test_index]
    
    model_xgb = xgb.sklearn.XGBClassifier(objective="binary:logistic", random_state=2020)
    model_xgb.fit(xtr, ytr)
    pred_test_xgb = model_xgb.predict(xvl)
    score_xgb.append(accuracy_score(yvl,pred_test_xgb))
    print('\nAccuracy_score : ',score_xgb[i-1])
    i+=1 
    
print("\nThe mean validation accuracy of the XGBoost model is : ", mean(score_xgb))

# Visualize the ROC curve

pred_xgb_prob=model_xgb.predict_proba(xvl)[:,1]

fpr, tpr, _ = metrics.roc_curve(yvl,  pred_xgb_prob)
auc = metrics.roc_auc_score(yvl, pred_xgb_prob)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="Validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4) 
plt.show()

importances=pd.Series(model_xgb.feature_importances_, index=X.columns) 
importances.plot(kind='barh', figsize=(12,8))


from sklearn.model_selection import GridSearchCV
# Provide range for max_depth from 2 to 20 with an interval of 2 
# and from 40 to 200 with an interval of 20 for n_estimators 

paramgrid_gb = {'learning_rate':[0.05, 0.1, 0.15], 'max_depth': list(range(3, 21, 3)), 'n_estimators': list(range(60, 160, 20))}

grid_search_gb=GridSearchCV(GradientBoostingClassifier(max_features='auto',random_state=2020),paramgrid_gb)

# Fit the grid search model 

grid_search_gb.fit(X_train,y_train)

# Estimating the optimized value 


grid_search_gb.best_estimator_



i=1 
# kf = KFold(n_splits=5,random_state=2020,shuffle=True) 

score_gb = []

for train_index,test_index in kf.split(X_train,y_train):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X_train[train_index],X_train[test_index]
    ytr,yvl = y_train.iloc[train_index],y_train.iloc[test_index]
    # model_gb = GradientBoostingClassifier(learning_rate=0.1, max_features='sqrt', max_depth=18, n_estimators=120, random_state=2020)
    
    model_gb = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                          learning_rate=0.15, loss='deviance', max_depth=3,
                                          max_features='auto', max_leaf_nodes=None,
                                          min_impurity_decrease=0.0, min_impurity_split=None,
                                          min_samples_leaf=1, min_samples_split=2,
                                          min_weight_fraction_leaf=0.0, n_estimators=80,
                                          n_iter_no_change=None, presort='deprecated',
                                          random_state=2020, subsample=1.0, tol=0.0001,
                                          validation_fraction=0.1, verbose=0,warm_start=False)
    
    model_gb.fit(xtr, ytr)
    pred_test_gb = model_gb.predict(xvl)
    score_gb.append(accuracy_score(yvl,pred_test_gb))
    print('\nAccuracy_score : ',score_gb[i-1])
    i+=1 
    
print("\nThe mean validation accuracy of the Gradient Boosting model is : ", mean(score_gb))

# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set

pred_gb = model_gb.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, pred_gb))
print()
print("Classification Report")
print(classification_report(y_val, pred_gb))
# Visualize the ROC curve

pred_gb_prob=model_gb.predict_proba(xvl)[:,1]

fpr, tpr, _ = metrics.roc_curve(yvl,  pred_gb_prob)
auc = metrics.roc_auc_score(yvl, pred_gb_prob)
plt.figure(figsize=(12,8))
plt.plot(fpr,tpr,label="Validation, auc="+str(auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4) 
plt.show()

final_test_df.shape
final_train_df.shape
final_test_df.columns
final_train_df.columns
final_test_df.head()
X_train
X_train.shape
train_columns
target_column
X_test = final_test_df[train_columns]
X_test.head()
X_test.shape
pred_gb_test = model_gb.predict(X_test)
pred_gb_prob_test=model_gb.predict_proba(X_test)[:,1]
pred_gb_prob_test.shape
pred_gb_prob_test
submission=pd.read_csv("../input/av-janata-hack-payment-default-prediction/sample_submission_gm6gE0l.csv")
submission
submission['default_payment_next_month']=pred_gb_prob_test
original_test_df['ID']
submission['ID']=original_test_df['ID']
submission.to_csv("CC_Payment_Default_Janata_Hack_31_May.csv", index=False)
