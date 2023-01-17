#import required packages

import pandas as pd

import matplotlib.pyplot as plt#load data sets



train = pd.read_csv('../input/train.csv')

transactions = pd.read_csv('../input/transactions.csv')
train.sample(5)  

train.info()
#change is_churn to categorical variable

train['is_churn'] = train['is_churn'].astype('category')
# take a look is_churn

train['is_churn'].value_counts().plot(kind='bar')

# As we can see, most members are renewal their memberships
#cross table to see exact number of is_churn

ischurn_tab = pd.crosstab(index=train['is_churn'],  # Make a crosstab

                              columns='Count')      # Name the count column

ischurn_tab
#Look at transactions data

transactions.sample(10)
transactions.info()

# 9 columns total 21,547,746 records
#examine msno

msno_distinct_id = transactions.drop_duplicates().msno.value_counts()
msno_distinct_id 

# As you can see, some user made more than one transaction
#merge two data sets 

train_transactions = pd.merge(train, transactions, on='msno')
train_transactions.sample(20)
#change is_auto_renew and is_cancel to categorical variables

transactions['is_auto_renew'] = transactions['is_auto_renew'].astype('category')

transactions['is_cancel'] = transactions['is_cancel'].astype('category')
# Table of is_churn vs is_auto_renew

ischurn_isautorenew = pd.crosstab(index=train_transactions['is_auto_renew'],  

                              columns=train_transactions['is_churn'],

                              margins=True)      

ischurn_isautorenew.columns = ['No','Yes', 'isChurnTotal']

ischurn_isautorenew.index = ['NoAutoRenew','YesAutoRenew', 'Total']

ischurn_isautorenew

ischurn_iscancel = pd.crosstab(index=train_transactions['is_cancel'],  

                              columns=train_transactions['is_churn'],

                              margins=True)    

ischurn_iscancel.index = ['NoCancel','YesCancel', 'Total']

ischurn_iscancel
