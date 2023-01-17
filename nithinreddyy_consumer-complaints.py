train1=r'D:\Downloads\Digital Nest\P1 Data\Consumer_Complaints_train.csv'

test1=r'D:\Downloads\Digital Nest\P1 Data\Consumer_Complaints_test_share.csv'
import numpy as np

import pandas as pd
train=pd.read_csv(train1)

test=pd.read_csv(test1)
test5=pd.read_csv(test1)
train.head()
test.head()
test['Consumer disputed?']=np.nan

test['data']='test'

train['data']='train'
all_data=pd.concat([train,test], axis=0)
test.shape
all_data.dtypes
all_data['ZIP code'].nunique()
all_data.groupby(['Company','Consumer disputed?']).value_counts()
k=all_data['Date received'].str.split("-",expand = True).astype(int)
all_data['Date received_int']=(k[0]*365)+(k[1]*30)+k[2]
all_data['Date received_int']
k=all_data['Date sent to company'].str.split("-",expand = True).astype(int)
all_data['Date sent to company_int']=((k[0]*365)+(k[1]*30)+k[2]).astype(int)
all_data['Date sent to company_DOW']=pd.to_datetime(all_data['Date sent to company']).dt.dayofweek.astype(int)
all_data['Date received_DOW']=pd.to_datetime(all_data['Date received']).dt.dayofweek.astype(int)
all_data['Date received_DOW']
all_data['Date sent to company_DOW']=pd.to_datetime(all_data['Date sent to company']).dt.dayofweek.astype(int)
all_data.drop(['Date received','Date sent to company'], axis=1, inplace=True)
all_data.drop(['Complaint ID','Consumer complaint narrative','ZIP code'], axis=1, inplace=True)
all_data.drop(['Company','Issue','State','Sub-issue','Sub-product'], axis=1, inplace=True)
all_data.dtypes
cat_vars=all_data.select_dtypes(['object']).columns
cat_vars
for col in cat_vars[:-1]:

    dummy=pd.get_dummies(all_data[col],drop_first=True,prefix=col)

    all_data=pd.concat([all_data,dummy],axis=1)

    del all_data[col]

    print(col)

del dummy
all_data.dtypes
all_data.isnull().sum()
train=all_data[all_data['data']=='train']

test=all_data[all_data['data']=='test']
del train['data']
test.drop(['data','Consumer disputed?_Yes'],axis=1,inplace=True)
train.shape
test.shape
x_train=train.drop('Consumer disputed?_Yes',axis=1)

y_train=train['Consumer disputed?_Yes']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
param_dist = {"n_estimators":[100,200,300,500,700,1000], 

"max_features": [5,10,20,25,30,35], 

"bootstrap": [True, False], 

"class_weight":[None,'balanced'], 

"criterion":['entropy','gini'], 

"max_depth":[None,5,10,15,20,30,50,70], 

"min_samples_leaf":[1,2,5,10,15,20], 

"min_samples_split":[2,5,10,15,20]} 
rf=RandomForestClassifier()
n_iter_search=10



random_search = RandomizedSearchCV(rf, param_distributions=param_dist, 

n_iter=n_iter_search, 

scoring='roc_auc', 

cv=5, n_jobs=-1,verbose=20) 

random_search.fit(x_train, y_train)
rf_bt=random_search.best_estimator_
train_score=rf_bt.predict_proba(x_train)
cutoffs=np.linspace(0.01,0.99,99)

cutoffs
train_score=rf_bt.predict_proba(x_train)[:,1] # the predicted response variable values

real=y_train # the actual response variable values

print(rf_bt.classes_)
from sklearn.metrics import fbeta_score
KS_all=[]

for cutoff in cutoffs:

    predicted=(train_score>cutoff).astype(int)

    TP=((predicted==1) & (real==1)).sum()

    TN=((predicted==0) & (real==0)).sum()

    FP=((predicted==1) & (real==0)).sum()

    FN=((predicted==0) & (real==1)).sum()

    P=TP+FN

    N=TN+FP

    KS=(TP/P)-(FP/N)

    KS_all.append(KS)
list(zip(cutoffs,KS_all))
mycutoff=cutoffs[KS_all==max(KS_all)][0]

mycutoff # gives the cutoff value where KS is maximum
test_score=rf_bt.predict_proba(test)[:,1]

test_score
test_classes=(test_score>mycutoff).astype(int)
test5.head()
CustID=test5['Complaint ID']
pd.DataFrame(test_classes).to_csv("ShraddhaP1.csv",index=False)
ClsSvm=clf_svm
ClsSvm
train_score=clf_svm.predict(x_train)
train_score.sum()
test_score=clf_svm.predict(test)
test_classes.sum()
from sklearn.metrics import roc_curve, auc, roc_auc_score,accuracy_score
roc_auc_score(y_train,train_score)
test_classes=(test_score>mycutoff).astype(int)
target=pd.DataFrame(test_classes)
target.dtypes
target=target.replace(to_replace=[0,1], value=['No', 'Yes'])
final=pd.concat([test5['Complaint ID'],target],axis=1)
test_classes
final=final.rename(columns={0:'Consumer disputed?'})
final
final=pd.concat([CustID,target])
pd.DataFrame(final).to_csv("Nithin_Predicted.csv",index=False)