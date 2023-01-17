import pandas as pd

import numpy as np

import category_encoders as ce

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from xgboost import XGBClassifier
train = pd.read_csv("../input/racism1/Train.csv")

test= pd.read_csv("../input/racism1/Test.csv")

sample = pd.read_csv("../input/racism1/sample_submission.csv")

mask = pd.read_csv("../input/racism1/unlinked_masked_final.csv")

#variabs = pd.read_csv('../input/racism1/VariableDefinitions.csv')
train.TransactionStartTime=pd.to_datetime(train.TransactionStartTime)

test.TransactionStartTime=pd.to_datetime(test.TransactionStartTime)

train.IssuedDateLoan=pd.to_datetime(train.IssuedDateLoan)

test.IssuedDateLoan=pd.to_datetime(test.IssuedDateLoan)

train.PaidOnDate=pd.to_datetime(train.PaidOnDate)

train.DueDate=pd.to_datetime(train.DueDate)
train['mean_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').mean().AmountLoan)

train['max_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').max().AmountLoan)

train['min_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').min().AmountLoan)

train['std_loan_cus']=train['CustomerId'].map(train.groupby('CustomerId').std().AmountLoan)



test['mean_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').mean().AmountLoan)

test['max_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').max().AmountLoan)

test['min_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').min().AmountLoan)

test['std_loan_cus']=test['CustomerId'].map(train.groupby('CustomerId').std().AmountLoan)
train['Number_Of_Split_Payments'] = 0 

test['Number_Of_Split_Payments']=0
train['Number_Of_Split_Payments']=train['TransactionId'].map(train.groupby('TransactionId').count()['Number_Of_Split_Payments'])

test['Number_Of_Split_Payments']=test['TransactionId'].map(test.groupby('TransactionId').count()['Number_Of_Split_Payments'])
train.groupby('TransactionId').CustomerId.nunique()[train.groupby('TransactionId').CustomerId.nunique()>1]
train.drop(train[(train['TransactionId'] =='TransactionId_703')|((train['TransactionId'] =='TransactionId_927'))].index,axis=0,inplace=True)
train.drop_duplicates(subset=['TransactionId'],keep='last',inplace=True)

test.drop_duplicates(subset=['TransactionId'],keep='last',inplace=True)
train.drop(['CountryCode','Currency','CurrencyCode','SubscriptionId','ProviderId','ChannelId'],axis=1,inplace=True)

test.drop(['CountryCode','CurrencyCode','SubscriptionId','ProviderId','ChannelId'],axis=1,inplace=True)
train['Count_Rejected_Loans'] = train['CustomerId'].map(train[train.TransactionStatus==0].groupby('CustomerId').LoanId.size())

test['Count_Rejected_Loans'] = test['CustomerId'].map(train[train.TransactionStatus==0].groupby('CustomerId').LoanId.size())

train.Count_Rejected_Loans.fillna(value=0,inplace=True)

test.Count_Rejected_Loans.fillna(value=0,inplace=True)
## объединяем

all_data=pd.concat((train,test)).copy()

## 

all_data['Cumulative_Reject']=0

all_data.loc[all_data.TransactionStatus==0,'Cumulative_Reject'] = all_data[all_data.TransactionStatus==0].groupby('CustomerId').cumcount()

## Возвращаем в тест и трейн

train1=all_data[:len(train)]

test1=all_data[len(train):]

train['Cumulative_Reject']=0

test['Cumulative_Reject']=0

train['Cumulative_Reject']=train1['Cumulative_Reject']

test['Cumulative_Reject']=test1['Cumulative_Reject']
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
train=train[train.IsDefaulted.notnull()]
train.loc[:,'new_customer']=0

test.loc[:,'new_customer']=0

train.loc[train.mean_cus_transac.isnull(),'new_customer']=1

test.loc[test.mean_cus_transac.isnull(),'new_customer']=1
#train.loc[:,'DefaultedProductId'] = train['ProductId'].map(train.groupby('ProductId').sum()['IsDefaulted'])

#test.loc[:,'DefaultedProductId'] = train['ProductId'].map(train.groupby('ProductId').sum()['IsDefaulted'])
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

            #'LoanId', , 'LoanApplicationId', 'ThirdPartyId',

            #'Number_Of_Split_Payments', 

            #'Count_Rejected_Loans', 

            #'Cumulative_Reject',

           #'prchs_mean', #'prchs_std',

           'prchs_max', 

           'prchs_min', #'month',

            #'InvestorId',

            #'Day_Of_Week',

           'Day_in_month',#'mean_loan_cus',

            #'max_loan_cus',

            #'min_loan_cus',

            #'new_customer',

            'before_due_mean', 'before_due_std',

           'before_due_min', 'before_due_max', 

            'count_cus_transac',

                #'DefaultedProductId'

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

from sklearn.model_selection import GridSearchCV

parameters = {'n_estimators': [25,50],'max_depth': [2,3,4,5,6,7,8,9],'min_samples_split': [2]}

xgb_cv = GridSearchCV(xgb,param_grid=parameters,cv=5,n_jobs=-1)

xgb_cv.fit(X,y)

test_pred=xgb_cv.predict_proba(X_test)[:,1]

#test_pred=xgb.predict(X_test)
#train1 = train.fillna(0)

#X = train1[features]
#X
#from sklearn.feature_selection import RFE



#rfe = RFE(xgb, 8)



#fit = rfe.fit(X, train1.IsDefaulted)
#X.head()
#u = fit.support_



#fit.ranking_
#df1 = pd.DataFrame(X.T[u])

#print(df1)
#X.shape
#X.head()
#from sklearn.feature_selection import SelectFromModel

#from sklearn.svm import LinearSVC



#lsvc = LinearSVC(C=0.001, penalty="l1", dual=False).fit(X, train1.IsDefaulted)

#model = SelectFromModel(lsvc, prefit=True)

#X_new = model.transform(X)
#X_new.shape
#df = pd.DataFrame(X_new)

#df
xgb_cv.best_estimator_
sample_submission = pd.DataFrame(columns=['TransactionId','IsDefaulted'])

sample_submission['TransactionId'] = test['TransactionId']

sample_submission['IsDefaulted'] = test_pred
sample_submission.to_csv('Submission.csv',index=False)