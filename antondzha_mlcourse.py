## importing libraries

import pandas as pd

import numpy as np

import category_encoders as ce

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
## reading the files and loading them into dataframes.

train = pd.read_csv("../input/creditscoringdata/Train.csv")

test= pd.read_csv("../input/creditscoringdata/Test.csv")

sample = pd.read_csv("../input/creditscoringdata/sample_submission.csv")

mask = pd.read_csv("../input/creditscoringdata/unlinked_masked_final.csv")

variabs = pd.read_csv('../input/creditscoringdata/VariableDefinitions.csv')
## Transform dates types from 'object' to 'datetime'

train.TransactionStartTime=pd.to_datetime(train.TransactionStartTime)

test.TransactionStartTime=pd.to_datetime(test.TransactionStartTime)

train.IssuedDateLoan=pd.to_datetime(train.IssuedDateLoan)

test.IssuedDateLoan=pd.to_datetime(test.IssuedDateLoan)

train.PaidOnDate=pd.to_datetime(train.PaidOnDate)

train.DueDate=pd.to_datetime(train.DueDate)
train.corr()
#train['mean_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['mean'])#

train['mean_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').mean().AmountLoan)

#train['max_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['max'])

train['max_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').max().AmountLoan)

#train['min_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['min'])

train['min_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').min().AmountLoan)

#train['std_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['std'])

train['std_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').std().AmountLoan)



#test['mean_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['mean'])

test['mean_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').mean().AmountLoan)

#test['max_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['max'])

test['max_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').max().AmountLoan)

#test['min_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['min'])

test['min_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').min().AmountLoan)

#test['std_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').agg(('mean','std','min','max')).AmountLoan['std'])

test['std_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').std().AmountLoan)
## creating variables to transfer the information contained in the rows of the same transaction.

train['Number_Of_Split_Payments'] = 0 ## this is a count on the number of payments on the same loan. It will take a 0 for singled-rowed transactions, 1+ for multi-row transacs.

#train['Sum_Diff_Time_Payments'] = 0 ## I'm thinking of summing the delays between all payments made on a loan. It will take 0 for loans paid in a single time, 1+ for multiple payments on the same loan.

test['Number_Of_Split_Payments']=0

#test['Sum_Diff_Time_Payments']=0
## creating the feature : number of split payments on a loan.

#train['Number_Of_Split_Payments']=train['TransactionId'].map(train.groupby('TransactionId').agg('count')['Number_Of_Split_Payments'])

train['Number_Of_Split_Payments']=train['TransactionId'].map(train.groupby('TransactionId').count()['Number_Of_Split_Payments'])

#test['Number_Of_Split_Payments']=test['TransactionId'].map(test.groupby('TransactionId').agg('count')['Number_Of_Split_Payments'])

test['Number_Of_Split_Payments']=test['TransactionId'].map(test.groupby('TransactionId').count()['Number_Of_Split_Payments'])
train.drop(train[(train['TransactionId'] =='TransactionId_703')|((train['TransactionId'] =='TransactionId_927'))].index,axis=0,inplace=True)
## Lets drop the duplicate rows with the same transaction ID and keep the last one. (as in with the latest payment installment )

train.drop_duplicates(subset=['TransactionId'],keep='last',inplace=True)

test.drop_duplicates(subset=['TransactionId'],keep='last',inplace=True)
train.drop(['CountryCode','Currency','CurrencyCode','SubscriptionId','ProviderId','ChannelId'],axis=1,inplace=True)

test.drop(['CountryCode','CurrencyCode','SubscriptionId','ProviderId','ChannelId'],axis=1,inplace=True)
train['Count_Rejected_Loans'] = train['CustomerId'].map(train[train.TransactionStatus==0].groupby('CustomerId').LoanId.size())

test['Count_Rejected_Loans'] = test['CustomerId'].map(train[train.TransactionStatus==0].groupby('CustomerId').LoanId.size())

## then we should impute the columns of customers that were not found in the rejected list with 0 as in they have never been rejected.

train.Count_Rejected_Loans.fillna(value=0,inplace=True)

test.Count_Rejected_Loans.fillna(value=0,inplace=True)
## group train/test together to perform cumulative count

all_data=pd.concat((train,test)).copy()

## Initialize and compute values for the new feature

all_data['Cumulative_Reject']=0

all_data.loc[all_data.TransactionStatus==0,'Cumulative_Reject'] = all_data[all_data.TransactionStatus==0].groupby('CustomerId').cumcount()

## Separate all_data into train and test

train1=all_data[:len(train)]

test1=all_data[len(train):]

train['Cumulative_Reject']=0

test['Cumulative_Reject']=0

train['Cumulative_Reject']=train1['Cumulative_Reject']

test['Cumulative_Reject']=test1['Cumulative_Reject']

all_data
purchasestats=train[train.TransactionStatus==0].groupby('CustomerId').Value.agg(('mean','std','min','max'))

train['prchs_mean']=train['CustomerId'].map(purchasestats['mean'])

train['prchs_std']=train['CustomerId'].map(purchasestats['std'])

train['prchs_max']=train['CustomerId'].map(purchasestats['max'])

train['prchs_min']=train['CustomerId'].map(purchasestats['min'])

test['prchs_mean']=test['CustomerId'].map(purchasestats['mean'])

test['prchs_std']=test['CustomerId'].map(purchasestats['std'])

test['prchs_max']=test['CustomerId'].map(purchasestats['max'])

test['prchs_min']=test['CustomerId'].map(purchasestats['min'])
valuegroups=mask.groupby('CustomerId').Value.agg(('mean','std','min','max','count'))

train['mean_cus_transac']=train['CustomerId'].map(valuegroups['mean'])

train['std_cus_transac']=train['CustomerId'].map(valuegroups['std'])

train['min_cus_transac']=train['CustomerId'].map(valuegroups['min'])

train['max_cus_transac']=train['CustomerId'].map(valuegroups['max'])

train['count_cus_transac']=train['CustomerId'].map(valuegroups['count'])

test['mean_cus_transac']=test['CustomerId'].map(valuegroups['mean'])

test['std_cus_transac']=test['CustomerId'].map(valuegroups['std'])

test['min_cus_transac']=test['CustomerId'].map(valuegroups['min'])

test['max_cus_transac']=test['CustomerId'].map(valuegroups['max'])

test['count_cus_transac']=test['CustomerId'].map(valuegroups['count'])
train['Day_Of_Week']= train.TransactionStartTime.dt.weekday

test['Day_Of_Week'] =test.TransactionStartTime.dt.weekday

train['Day_in_month']=train.TransactionStartTime.dt.day

test['Day_in_month']=test.TransactionStartTime.dt.day
from datetime import date

datemin = date(2018,9,21)

datemax= date(2019,7,17)

(datemax-datemin).days

datesinc=pd.DataFrame(columns=['date','inc_value'])

datesinc.loc[0,'inc_value']=1

datesinc.loc[0,'date']=datemin

from datetime import timedelta

for i in range(2,301):

    datesinc.loc[i-1,'inc_value']=i

    datesinc.loc[i-1,'date']=datemin + timedelta(days=i-1)

train['inc_value_date']=train.TransactionStartTime.dt.date.map(datesinc.set_index('date').inc_value)

test['inc_value_date']=test.TransactionStartTime.dt.date.map(datesinc.set_index('date').inc_value)
train.inc_value_date = train.inc_value_date.astype(np.int64)

test.inc_value_date = test.inc_value_date.astype(np.int64)
aa=train[(train.TransactionStatus==1)&(train.TransactionStartTime<train.DueDate)].groupby('CustomerId').agg(('count','mean','std','min','max')).Value

#train['number_transac_before_due']=train['CustomerId'].map(aa['count'])

train['before_due_mean'] = train['CustomerId'].map(aa['mean'])

train['before_due_std'] = train['CustomerId'].map(aa['std'])

train['before_due_min'] = train['CustomerId'].map(aa['min'])

train['before_due_max'] = train['CustomerId'].map(aa['max'])

test['before_due_mean'] = test['CustomerId'].map(aa['mean'])

test['before_due_std'] = test['CustomerId'].map(aa['std'])

test['before_due_min'] = test['CustomerId'].map(aa['min'])

test['before_due_max'] = test['CustomerId'].map(aa['max'])
train[train.IsDefaulted==1].describe()
train=train[train.IsDefaulted.notnull()]
train.loc[:,'new_customer']=0

test.loc[:,'new_customer']=0

train.loc[train.mean_cus_transac.isnull(),'new_customer']=1

test.loc[test.mean_cus_transac.isnull(),'new_customer']=1
train
features = [#'CustomerId', #'TransactionStartTime', 

            'Value', #'Amount',

       #'TransactionId', 

            #'BatchId', 

              #'ProductId',

    'mean_cus_transac',

    #'std_cus_transac', 

    'min_cus_transac', 

    'max_cus_transac', 

       #'ProductCategory', #'TransactionStatus', 

            #'IssuedDateLoan',

    #'mean_value_on_missed',

       #'LoanId', , 'LoanApplicationId', 'ThirdPartyId',

       #'Number_Of_Split_Payments', 

        #    'Count_Rejected_Loans', 

           #'Cumulative_Reject',

       #'prchs_mean', 'prchs_std',

    'prchs_max', 

    'prchs_min', #'month',

         #   'InvestorId',

            #'Day_Of_Week',

    'Day_in_month',#'mean_loan_cus',

    #'max_loan_cus',

    #'min_loan_cus',

    'new_customer',

    #'inc_value_date'

    #'before_due_mean'#, 'before_due_std',

       'before_due_min', 'before_due_max', 

            'count_cus_transac'#,'Cnt_missed_payment'

]
from sklearn.model_selection import train_test_split
X = train[features]

X_test = test[features]

y=train.IsDefaulted.copy()
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,stratify=y,random_state=10)
xgb = XGBClassifier(max_depth=4,colsample_bytree=0.6,min_child_weight=10,learning_rate=0.25,n_estimators=100,objective = "binary:logistic",random_state=26)

xgb.fit(X_train,y_train)

val_pred=xgb.predict_proba(X_val)[:,1]

#val_pred=xgb.predict(X_val)

print(roc_auc_score(y_val,val_pred))

X_train
xgb.fit(X,y)

test_pred=xgb.predict_proba(X_test)[:,1]

#test_pred=xgb.predict(X_test)
sample_submission = pd.DataFrame(columns=['TransactionId','IsDefaulted'])

sample_submission['TransactionId'] = test['TransactionId']

sample_submission['IsDefaulted'] = test_pred
sample_submission.to_csv('Submission.csv',index=False)
X