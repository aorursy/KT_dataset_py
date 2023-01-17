import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.figure_factory as ff

from pandas import DataFrame

import copy

import re

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

import statsmodels.api as sm

import time

import os

from statsmodels.discrete.discrete_model import Logit
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data=pd.read_excel('/kaggle/input/mobilechurndataxlsx/mobile-churn-data.xlsx')

data=data.drop('user_account_id',axis=1)
def summarizing(df):

    obs=df.shape[0] # the total number of observatinos 

    types=df.dtypes

    counts=df.apply(lambda x:x.count())

    unique=df.apply(lambda x:[x.unique()] ).transpose()

    distincts=df.apply(lambda x:x.unique().shape[0])

    null=df.isnull().sum()

    missing_rate=(round(df.isnull().sum()/obs*100,2)).astype('str')+'%'

    skew=df.skew()

    kurtosis=df.kurtosis()

    table=pd.concat([types,counts,unique,distincts,null,missing_rate,skew,kurtosis],axis=1)

    table.columns=['Data_Types','Number_of_NonNull','Unique_Values','Number_of_Unique Values','Missing_Number','Missing_Rate','Skew'

                ,'Kurtosis']

    print('The shapes of data:{0}'.format(df.shape))

    print('--'*30)

    print('Types Counts:\n',table.Data_Types.value_counts())

    print('--'*30)
summarizing(data)
display(data.describe().transpose())
#correlation with uuser_lifetime

display(data.corr()['user_lifetime'].sort_values(ascending=False))
#merging all input features with same infomration into one variable 

data['calls_outgoing_inactive_days']=data.calls_outgoing_inactive_days

data['sms_outgoing_inactive_days']=data.sms_outgoing_inactive_days



#Remove input features conating same information

removing_features=['calls_outgoing_to_onnet_inactive_days', 'calls_outgoing_to_offnet_inactive_days','calls_outgoing_to_abroad_inactive_days',       

'sms_outgoing_to_abroad_inactive_days','sms_outgoing_inactive_days','sms_outgoing_to_onnet_inactive_days','sms_outgoing_to_offnet_inactive_days']

data=data.drop(removing_features,axis=1)

data=data.rename(columns={'user_no_outgoing_activity_in_days':'min_outgoing_inactive_days'})

data.head()
## create another varialbe to indicate whether they have used the telecomunication service at leaste once in a month. 

data['user_has_outgoing']=(data.user_has_outgoing_calls+data.user_has_outgoing_sms).map(lambda x: 'yes' if x>0 else 'no')
result1=data.pivot_table(['min_outgoing_inactive_days','user_lifetime','user_spendings'],index='user_has_outgoing',aggfunc='mean')

result1
result1['life_time_ratio']=result1.user_lifetime/result1.min_outgoing_inactive_days*result1.user_spendings/30

result1
sns.set_style('white')

fig,ax=plt.subplots(figsize=(12,8))

ax2=ax.twinx()

sns.distplot(data[data.user_has_outgoing=='yes'].user_lifetime,color='b',ax=ax,label='active')

sns.distplot(data[data.user_has_outgoing=='no'].user_lifetime,color='coral',ax=ax2,label='inactive')

ax.set_xlabel('user_life_time',fontsize=15)

ax.set_title('Density plots of Lifetime of Two types of Users',fontsize=20)

ax.legend(loc='upper left',fontsize=14)

ax2.legend(loc='upper right',fontsize=14)

fig,ax=plt.subplots(figsize=(12,8))

sns.boxplot(x='user_has_outgoing',y='user_lifetime',data=data,ax=ax)

ax.set_title('Box plots of Lifetime of Two types of Users',fontsize=20)
result2=[['Returning Customers','Potentially Inactive','Total Counts of Both Groups','Proportion of total User After removal']]

a=data[(data.user_has_outgoing=='yes')&(data.user_lifetime>3000)].shape[0]

b=data[(data.user_has_outgoing=='no')&(data.user_lifetime<14000)].shape[0]



set1=(data.user_has_outgoing=='yes')&(data.user_lifetime<3000)

set2=(data.user_has_outgoing=='no')&(data.user_lifetime>14000)

c=data[set1^set2].shape[0]

d=str(round(c/data.shape[0]*100,2))+'%'



result2.append([a,b,c,d])

table=ff.create_table(result2)

table.layout.update(width=1100)

table.show()
data['user_type']=None

data.loc[(data.user_has_outgoing=='yes')&(data.user_lifetime<3000),'user_type']='likely_active_consumers'

data.loc[(data.user_has_outgoing=='yes')&(data.user_lifetime>3000),'user_type']='return_consumers'

data.loc[(data.user_has_outgoing=='no')&(data.user_lifetime<3000),'user_type']='possilbe_inactive_consumers'

data.loc[(data.user_has_outgoing=='no')&(data.user_lifetime>3000),'user_type']='highly_inactive'
fig,ax=plt.subplots(figsize=(14,5))

display(data.pivot_table('user_has_outgoing',index='user_type',columns='churn',aggfunc='count'))

data.pivot_table('user_has_outgoing',index='user_type',columns='churn',aggfunc='count').plot(kind='bar',ax=ax)

ax.set_title('Barplot for The Total Number of Four Type Users by Churn',fontsize=20)

ax.set_ylabel('number of users',fontsize=14)

ax.set_xlabel('user type',fontsize=14)
#to leave only on variable 'user has_outgoing'

has_columns=[]

for i in data.columns:

    if re.search('has',i):

        has_columns.append(i)

        

#drop the rest of columns related to 'has_outgoing' columns

data=data.drop(has_columns,axis=1)
data.corr()['reloads_sum'].sort_values(ascending=False)[1:4]
result4=data.pivot_table(['reloads_sum','user_account_balance_last','user_spendings'],index=['user_type','churn'],aggfunc='mean')

#reanme the label of churn ('no' for 0 and 'yes' for 1)

result4.index=result4.index.set_levels(['no','yes'],level=1)

result4=result4.reset_index()
result4=data.pivot_table(['reloads_sum','user_account_balance_last','user_spendings'],index=['user_type','churn'],aggfunc='mean')
from re import search

duration=[]

for i in data.columns:

    if search('duration',i):

        duration.append(i)

duration

result5=data.pivot_table(duration,index=['user_type','churn'],aggfunc='mean')

result5.index=result5.index.set_levels(['No','Yes'],level=1)

result5
gprs=[]

for i in data.columns:

    if search('gprs',i):

        gprs.append(i)



        result5=data.pivot_table('user_use_gprs',index=['user_type','churn'],aggfunc='count')

result5.index=result5.index.set_levels(['no','yes'],level=1)

result5=result5.reset_index()

result5=result5.rename(columns={'user_user_gprs':"number of user_gprs"})
gprs=gprs[1:]

result6=data.pivot_table(gprs,index=['user_type','churn'],aggfunc='mean')

result6.index=result6.index.set_levels(['no','yes'],level=1)

result6=result6.reset_index()

result6=result6.rename(columns={

 "gprs_inactive_days":"gprs_inactive_days(avg)",

"gprs_session_count":"gprs_session_count(avg)",

"gprs_spendings":"gprs_spendings(avg)",

"gprs_usage":"gprs_usage(avg)",

"last_100_gprs_usage":"last_100_gprs_usage(avg)"})
result5=result5.merge(result6,how='outer')

result5
g=sns.catplot(x='user_type',y='gprs_inactive_days(avg)',col='churn',kind='bar',data=result5)

g.set_xticklabels(rotation=30)

g.set_xlabels('user_type',fontsize=14)

g.set_ylabels('gprs_inactive_days(avg)',fontsize=14)
g=sns.catplot(x='user_type',y='gprs_spendings(avg)',col='churn',kind='bar',data=result5)

g.set_xticklabels(rotation=30)

g.set_xlabels('user_type',fontsize=14)

g.set_ylabels('gprs_spendings(avg)',fontsize=14)
g=sns.catplot(x='user_type',y='gprs_usage(avg)',col='churn',kind='bar',data=result5)

g.set_xticklabels(rotation=30)

g.set_xlabels('user_type',fontsize=14)

g.set_ylabels('gprs_usage(avg)',fontsize=14)
g=sns.catplot(x='user_type',y='last_100_gprs_usage(avg)',col='churn',kind='bar',data=result5)

g.set_xticklabels(rotation=30)

g.set_xlabels('user_type',fontsize=14)

g.set_ylabels('last_100_gprs_usage(avg)',fontsize=14)
result_corr=data.corr()['user_lifetime'].sort_values(ascending=False)

result_corr[gprs]
data1=data.copy()

data_response=data1.churn

data1=data1.drop(['churn','user_type'],axis=1)
conver_var=['user_intake','user_use_gprs','user_does_reload','user_does_reload'] 

def conversion(data):  

    for i in conver_var:

        data.loc[:,i]=data.loc[:,i].map(lambda x: 'no' if x==0 else 'yes')

    return data



data1=conversion(data1)



def quant_qualt_columns(x):

    quant=[]

    qualt=[]

    for i in x.columns:

        if x.loc[:,i].dtype=='int64' or x.loc[:,i].dtype=='float64':

            quant.append(i)

        else:

            qualt.append(i)

    return {0:quant,1:qualt}



quant_columns=quant_qualt_columns(data1)[0]

qualt_columns=quant_qualt_columns(data1)[1]
a=pd.DataFrame(scale(data1.loc[:,quant_columns]),columns=quant_columns)

b=data1.loc[:,qualt_columns]

data1=pd.concat([a,b],axis=1)



#since year is comprise of only zero, we will drop the feature 

data1=data1.drop('year',axis=1)
for var in qualt_columns:

    cat_list = pd.get_dummies(data1.loc[:,var],prefix=var)

    data1=data1.join(cat_list)

    

data_final=pd.concat([data1,data_response],axis=1)

data_final=data_final.drop(conver_var,axis=1)
X=data_final.loc[:,data_final.columns !='churn']

y=data_final.loc[:,'churn']
os=SMOTE(random_state=0)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

columns=X_train.columns



os_data_X,os_data_y=os.fit_sample(X_train,y_train)

os_data=pd.DataFrame(data=os_data_X,columns=columns)

os_data['churn']=os_data_y



#Check the numbers of our data

print("Length of oversampled data is                                            ",len(os_data_X))

print("Number of churn whose value is 0 in oversampled data                     ",len(os_data_y[os_data.churn==0]))

print("Number of chunr whose value is 1 in oversampled data in oversampled data ",len(os_data_y[os_data.churn==1]))
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

rfe = RFE(logreg, 40)

rfe = rfe.fit(os_data_X, os_data_y.ravel())

print(rfe.support_)

print(rfe.ranking_)
col_index=np.where(rfe.support_)

col=os_data.columns[col_index]

X=os_data.loc[:,col]

y=os_data.churn



col
tol = 0.0001

maxiter = 1000

DISP = 0





SOLVERS = ["newton", "nm","bfgs","lbfgs","powell","cg","ncg"] #,"basinhopping",]

for method in SOLVERS:

    t = time.time()

    model = Logit(y,X)

    result = model.fit(method=method, maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol,  # Hmmm.. needs to be reviewed.

                       disp=DISP)

    print("sm.Logit", method, time.time() - t)

    print("--------------------------------------------------------- ")



model = Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
result.summary()
# second summary

drop_col=['calls_outgoing_spendings','calls_outgoing_to_abroad_spendings','sms_outgoing_to_onnet_count','user_intake_no',

          'user_intake_yes','user_does_reload_yes']



X=X.drop(drop_col,axis=1)

model=Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
result.summary()
# Third Summary

drop_col=['reloads_sum','calls_outgoing_spendings_max','last_100_calls_outgoing_to_onnet_duration','last_100_sms_outgoing_to_offnet_count']

X=X.drop(drop_col,axis=1)

model=Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
result.summary()
#Fourth Try

drop_col=['calls_outgoing_to_offnet_count','last_100_calls_outgoing_to_abroad_duration']

X=X.drop(drop_col,axis=1)

model=Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
result.summary()
#Fifth Try



drop_col=['calls_outgoing_to_abroad_duration']

X=X.drop(drop_col,axis=1)

model=Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
#Sixth Try



drop_col=['calls_outgoing_to_onnet_spendings']

X=X.drop(drop_col,axis=1)

model=Logit(y,X)

result = model.fit(method='lbfgs',maxiter=maxiter,

                       niter=maxiter,

                       ftol=tol,

                       tol=tol, gtol=tol, pgtol=tol, 

                       disp=DISP)
result.summary()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix



confusion_matrix=confusion_matrix(y_test,y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve



logit_roc_auc=roc_auc_score(y_test,logreg.predict(X_test))

fpr,tpr,threshold=roc_curve(y_test,logreg.predict(X_test))

sns.set_style('whitegrid')

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()