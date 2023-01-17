# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pwd
pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')
pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')
train=pd.read_csv('../input/covid19-global-forecasting-week-2/train.csv')

test=pd.read_csv('../input/covid19-global-forecasting-week-2/test.csv')

submission=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')
train.isnull().sum()
train['Fatalities'].count()
train.count()
train.isnull().sum()
train['Fatalities'].value_counts()[1]
train['ConfirmedCases'].value_counts()[10]
train['Province_State'].isnull().sum()
train['Province_State'].describe()
#train.groupby(['Country_Region']).mean()
#train.groupby(['Country_Region','Province_State'],as_index=False).max().sum()
#train.groupby(['Country_Region','Province_State'],as_index=False).max().sort_values('ConfirmedCases',ascending=False)
#pd.set_option('display.max_columns',150)

#train.groupby(['Country_Region','Province_State'],as_index=False).max().sort_values('ConfirmedCases',ascending=False)
pd.set_option('display.max_rows',None)

train.groupby(['Country_Region','Province_State'],as_index=False).max().sort_values('ConfirmedCases',ascending=False)
pd.set_option('display.max_rows',None)

train.groupby(['Country_Region','Province_State'],as_index=False).max().sort_values('Fatalities',ascending=False)
#pd.set_option('display.max_rows',None)

#train.groupby(['Date','Country_Region']).max()
#pd.set_option('display.max_rows',None)

#train.groupby(['Date','Country_Region']).max().drop(['Id'],axis=1)
#train.shape
#submission.shape
#test.shape
display(submission.head())

display(test.head())

display(train.head())
len(test['Country_Region'].unique())
len(train['Country_Region'].unique())
len(test['Province_State'].unique())
len(train['Province_State'].unique())
display(test.head())

display(test.tail())

display(submission.head())

display(submission.tail())
train1=train[(train['Country_Region']=='US')&(train['Province_State']=='New York')]

display(train1)
train2=train[(train['Country_Region']=='China')&(train['Province_State']=='Hubei')]

display(train2)
train3=train[(train['Country_Region']=='US')&(train['Province_State']=='Washington')]

display(train3)
train4=train[(train['Country_Region']=='US')&(train['Province_State']=='New Jersey')]

display(train4)

train5=train[(train['Country_Region']=='US')&(train['Province_State']=='Louisiana')]

display(train5)
import matplotlib.pyplot as plt

plt.title('Louisiana COVID-19',color='red',size=20,style='italic',family='fantasy')

plt.hist(train5['Date'])

plt.xlabel('Date',color='red',size=16)
confirmed_total_date=train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date=train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date=confirmed_total_date.join(fatalities_total_date)



fig,(ax1,ax2)=plt.subplots(1,2,figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases",size=13)

ax1.set_ylabel("Number of cases",size=13)

ax1.set_xlabel("Date",size=13)



fatalities_total_date.plot(ax=ax2,color='orange')

ax2.set_title("Global decreased cases",size=13)

ax2.set_ylabel("Number of cases",size=13)

ax2.set_xlabel("Date",size=13)



display(confirmed_total_date)

display(fatalities_total_date)

display(total_date)
confirmed_total_data_noChina=train[train['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina=train[train['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina=confirmed_total_data_noChina.join(fatalities_total_date_noChina)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China",size=13)

ax1.set_ylabel("Number of cases",size=13)

ax1.set_xlabel("Date",size=13)

fatalities_total_date_noChina.plot(ax=ax2,color='orange')
confirmed_total_data_italy_2=train[(train['Country_Region']=='Italy')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_italy_2=train[(train['Country_Region']=='Italy')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_italy_2=confirmed_total_data_italy_2.join(fatalities_total_date_italy_2)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

total_date_italy_2.plot(ax=ax1)

ax1.set_title("Global confirmed cases of Italy",size=13)

ax1.set_ylabel("Number of cases",size=13)

ax1.set_xlabel("Date from the first confirmed case",size=13)

fatalities_total_date_italy_2.plot(ax=ax2,color='orange')
display(confirmed_total_data_italy_2.head())
confirmed_total_data_japan_2=train[(train['Country_Region']=='Japan')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_japan_2=train[(train['Country_Region']=='Japan')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_japan_2=confirmed_total_data_japan_2.join(fatalities_total_date_japan_2)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

total_date_japan_2.plot(ax=ax1)

ax1.set_title("Global confirmed cases of Japan",size=13)

ax1.set_ylabel("Number of cases",size=13)

ax1.set_xlabel("Date from the first confirmed case",size=13)

fatalities_total_date_japan_2.plot(ax=ax2,color='orange')
confirmed_total_data_afgan_2=train[(train['Country_Region']=='Afghanistan')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_afgan_2=train[(train['Country_Region']=='Afghanistan')&(train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_afgan_2=confirmed_total_data_afgan_2.join(fatalities_total_date_afgan_2)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))

total_date_afgan_2.plot(ax=ax1)

ax1.set_title("Global confirmed cases of Afghanistan",size=13)

ax1.set_ylabel("Number of cases",size=13)

ax1.set_xlabel("Date from the first confirmed case",size=13)

fatalities_total_date_afgan_2.plot(ax=ax2,color='orange')
submission.head()
display(train.head())

display(test.head())
train_week1=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

test_week1=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')
display(train_week1.head())

display(test_week1.head())
train_week1_extract=train_week1[['Country/Region','Lat','Long']]
train_week1_extract.columns=['Country_Region','Lat','Long']

train_week1_extract.head()
train_new=pd.merge(train,train_week1_extract,how='inner',on='Country_Region')
train_new.head()
train_new.isnull().sum()
train_new['Date']=train_new['Date'].apply(lambda x: x.replace("-",""))

train_new['Date']=train_new['Date'].astype(int)

train_new.head()
train_new=train_new.drop(['Province_State'],axis=1)
test.head()
test_new=pd.merge(test,train_week1_extract,how='inner',on='Country_Region')

test_new=test_new.drop(['Province_State'],axis=1)

test_new['Date']=test_new['Date'].apply(lambda x: x.replace("-",""))

test_new.head()
test_new.isnull().sum()
x=train_new[['Lat','Long','Date']]

y1=train_new[['ConfirmedCases']]

y2=train_new[['Fatalities']]

x_test=test_new[['Lat','Long','Date']]
from sklearn.ensemble import RandomForestClassifier

Tree_model=RandomForestClassifier(max_depth=4,random_state=0)
#Tree_model.fit(x,y1)

#pred1=Tree_model.predict(x_test)

#pred1=pd.DataFrame(pred1)

#pred1.columns=['ConfirmedCases']
#display(pred1.head())

#Tree_model.fit(x,y2)

#pred2=Tree_model.predict(x_test)

#pred2=pd.DataFrame(pred2)

#pred2.columns=['Fatalities']
import lightgbm as lgb
from sklearn import tree

from sklearn.model_selection import train_test_split

X_new=train_new[['Lat','Long','Date','ConfirmedCases','Fatalities']]

train_set,test_set=train_test_split(X_new,test_size=0.2,random_state=4)

drop_col=['ConfirmedCases','Fatalities']

X_train=train_set.drop(drop_col,axis=1)

y1_train=train_set['ConfirmedCases']

y2_train=train_set['Fatalities']

X_test=test_set.drop(drop_col,axis=1)

y1_test=test_set['ConfirmedCases']

y2_test=test_set['Fatalities']



lgb_train_1=lgb.Dataset(X_train,y1_train)

lgb_eval_1=lgb.Dataset(X_test,y1_test)



params={'metric':'rmse','max_depth':9}
gbm_1=lgb.train(params,

               lgb_train_1,

               valid_sets=(lgb_train_1,lgb_eval_1),

               num_boost_round=10000,

               early_stopping_rounds=100,

               verbose_eval=50)
predicted_1=gbm_1.predict(X_test)

predicted_1
display(X_test.head())

display(x_test.head())
x_test['Date']=x_test['Date'].astype(int)
predicted_1=gbm_1.predict(x_test)
predicted_1=pd.DataFrame(predicted)

predicted_1.columns=['ConfirmedCases']
predicted_1.head()
predicted_1.describe()
lgb_train_2=lgb.Dataset(X_train,y2_train)

lgb_eval_2=lgb.Dataset(X_test,y2_test)

gbm_2=lgb.train(params,

               lgb_train_2,

               valid_sets=(lgb_train_2,lgb_eval_2),

               num_boost_round=10000,

               early_stopping_rounds=100,

               verbose_eval=50)

predicted_2=gbm_2.predict(x_test)

predicted_2=pd.DataFrame(predicted)

predicted_2.columns=['Fatalities']

predicted_2.head()
submission.head()
sub_new=submission[['ForecastId']]

sub_new
submit=pd.concat([sub_new,predicted_1,predicted_2],axis=1)

submit.head(50)
submission.to_csv('submission.csv',index=False)