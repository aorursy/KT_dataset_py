#data loading
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('../input/bank.csv')
data.head()
data.shape
#check for missing values
data.isnull().sum()
deposit=data['deposit'].value_counts()
deposit.plot(kind='bar')
plt.title('Ratio of acceptance and rejection')
ageyes=data['age'][data.deposit=='yes']
ageno=data['age'][data.deposit=='no']
plt.figure()
ageyes.plot(kind='hist',color='green')
plt.title('Age Distrubution Accepted Customer')
plt.figure()
ageno.plot(kind='hist',color='gray')
plt.title('Age Distrubution Rejected Customer')
plt.show()
jobyes=data['job'][data.deposit=='yes'].value_counts()
jobno=data['job'][data.deposit=='no'].value_counts()
job=pd.concat([jobyes,jobno],axis=1)
job.columns=['yes','no']
job.plot(kind='bar')

msyes=data['marital'][data.deposit=='yes'].value_counts()
msno=data['marital'][data.deposit=='no'].value_counts()
ms=pd.concat([msyes,msno],axis=1)
ms.columns=['accepted','rejected']
print(ms)
ms.plot(kind='bar')
educationyes=data['education'][data.deposit=='yes'].value_counts()
educationno=data['education'][data.deposit=='no'].value_counts()
education=pd.concat([educationyes,educationno],axis=1)
education.columns=['yes','no']
education.plot(kind='bar')
default_yes=data['default'][data.deposit=='yes'].value_counts()
default_no=data['default'][data.deposit=='no'].value_counts()
default=pd.concat([default_yes,default_no],axis=1)
default.columns=['accepted','rejected']
default.plot(kind='bar')
balance_yes=data['balance'][data.deposit=='yes']
balance_no=data['balance'][data.deposit=='no']
plt.figure()
balance_yes.plot(kind='hist',color='green')
plt.title('Distrubtion of balance of accepted customer')
plt.figure()
balance_no.plot(kind='hist',color='red')
plt.title('Distrubtion of balance of rejected customer')

housing_yes=data['housing'][data.deposit=='yes'].value_counts()
housing_no=data['housing'][data.deposit=='no'].value_counts()
housing=pd.concat([housing_yes,housing_no],axis=1)
plt.figure()
housing.columns=['accepted','rejected']
housing.plot(kind='bar')
plt.xlabel('Housing')

loan_yes=data['loan'][data.deposit=='yes'].value_counts()
loan_no=data['loan'][data.deposit=='no'].value_counts()
loan=pd.concat([loan_yes,loan_no],axis=1)
plt.figure()
loan.columns=['accepted','rejected']
loan.plot(kind='bar')
plt.xlabel('loan')
data['contact'].unique()
contact_yes=data['contact'][data.deposit=='yes'].value_counts()
contact_no=data['contact'][data.deposit=='no'].value_counts()
contact=pd.concat([contact_yes,contact_no],axis=1)
plt.figure()
contact.columns=['yes','no']
contact.plot(kind='bar')

data['day'].unique()
day_yes=data['day'][data.deposit=='yes'].value_counts()
day_no=data['day'][data.deposit=='no'].value_counts()
day=pd.concat([day_yes,day_no],axis=1)
plt.figure(figsize=(10,20))
day.columns=['yes','no']
day.plot(kind='bar')
plt.show()

month_yes=data['month'][data.deposit=='yes'].value_counts()
month_no=data['month'][data.deposit=='no'].value_counts()
month=pd.concat([month_yes,month_no],axis=1)
plt.figure(figsize=(10,20))
month.columns=['yes','no']
month.plot(kind='bar')
plt.show()
data['duration'].unique()
duration_yes=data['duration'][data.deposit=='yes']
duration_no=data['duration'][data.deposit=='no']
plt.figure()
duration_yes.plot(kind='hist',color='green')
duration_no.plot(kind='hist',color='red')
plt.legend(['yes','no'])
campaign_yes=data['campaign'][data.deposit=='yes']
campaign_no=data['campaign'][data.deposit=='no']
plt.figure()
campaign_yes.plot(kind='hist',color='green')
plt.figure()
campaign_no.plot(kind='hist',color='red')

data['campaign'].unique()
data['pdays'].unique()
pdays_yes=data['pdays'][data.deposit=='yes']
pdays_no=data['pdays'][data.deposit=='no']
pdays_yes.plot(kind='hist',color='green')
pdays_no.plot(kind='hist',color='red')
previous_yes=data['previous'][data.deposit=='yes']
previous_no=data['previous'][data.deposit=='no']
plt.figure()
previous_yes.plot(kind='hist')
previous_no.plot(kind='hist')
plt.legend(['yes','no'])

data['poutcome'].unique()
poutcome_yes=data['poutcome'][data.deposit=='yes'].value_counts()
poutcome_no=data['poutcome'][data.deposit=='no'].value_counts()
poutcome=pd.concat([poutcome_yes,poutcome_no],axis=1)
poutcome.columns=['yes','no']
poutcome.plot(kind='bar')

