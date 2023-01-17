import numpy as np

import pandas as pd

import pandas

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import KFold,GroupKFold
train=pd.read_csv('../input/ltfs-2/train_fwYjLYX.csv')

test=pd.read_csv('../input/ltfs-2/test_1eLl9Yf.csv')
def mean_absolute_percentage_error(y_true, y_pred): 

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
train.head()
train['application_date']=pd.to_datetime(train['application_date'])

test['application_date']=pd.to_datetime(test['application_date'])
import holidays

hol_list = holidays.IND(years = [2017,2018,2019])

hol_list = [date for date,name in hol_list.items()]

train['hol'] = train['application_date'].isin(hol_list) * 1

test['hol'] = test['application_date'].isin(hol_list) * 1
def dateFeatures(df, label=None,seg=None):

    features = ['day','week','dayofweek','month','quarter','year','dayofyear','weekofyear','is_month_start','is_month_end','is_quarter_start','is_quarter_end','is_year_start','is_year_end']

    date = df['application_date']

    for col in features:

        df[col] = getattr(date.dt,col) * 1
train = train[['application_date','segment','case_count']]

train_s1=train[train['segment']==1].groupby(['application_date']).sum().reset_index().sort_values('application_date')

train_s2=train[train['segment']==2].groupby(['application_date']).sum().reset_index().sort_values('application_date')

test_s1=test[test['segment']==1][['application_date']].sort_values('application_date')

test_s2=test[test['segment']==2][['application_date']].sort_values('application_date')
dateFeatures(train_s1)

dateFeatures(train_s2)

dateFeatures(test_s1)

dateFeatures(test_s2)
test_s2.head()
sns.boxplot(train_s1['case_count'])
sns.distplot(train_s1['case_count'])
train_s1['case_count'].describe()
case_max = train_s1['case_count'].max()

train_s1[train_s1['case_count']==case_max]
train_s1[(train_s1['application_date'] >= '2017-03-01') & (train_s1['application_date'] <= '2017-03-31')]
train_s1[(train_s1['application_date'] >= '2018-03-01') & (train_s1['application_date'] <= '2018-03-31')]
train_s1[(train_s1['application_date'] >= '2019-03-01') & (train_s1['application_date'] <= '2019-03-31')]
train_s1[train_s1['case_count'] > 7000]
train_s1[train_s1['case_count']<20]
train_s1 = train_s1[(train_s1['case_count'] > 20) & (train_s1['case_count'] < 7000)]
train_s1 = train_s1[train_s1['case_count']<=10000]

train_s1=train_s1.reset_index().drop('index',axis=1)
train_s2.describe()
sns.distplot(train_s2['case_count'])
sns.boxplot(train_s2['case_count'])
train_s2[train_s2['case_count']>35000]
train_s2 = train_s2[train_s2['case_count']<36000]

train_s2=train_s2.reset_index().drop('index',axis=1)
y1 = train_s1['case_count']

y2 = train_s2['case_count']

train_s1.drop(['case_count','segment','application_date'],axis=1,inplace=True)

train_s2.drop(['case_count','segment','application_date'],axis=1,inplace=True)
test_s1.drop(['application_date'],axis=1,inplace=True)

test_s2.drop(['application_date'],axis=1,inplace=True)
kf=GroupKFold(n_splits=20)

s1models = []

s2models = []



X = train_s1

y = y1

loss = []



print("loss:")

grp = train_s1['day'].values

for train_index, test_index in kf.split(X,y,grp):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model=RandomForestRegressor(n_estimators = 150 ,random_state=42,max_features =8)

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    print(mean_absolute_percentage_error(y_test,preds))

    loss.append(mean_absolute_percentage_error(y_test,preds))

    s1models.append(model.predict(test_s1))
s1models = s1models[1:19]
X = train_s2

y = y2

loss = []

print("loss : ")

grp = train_s2['dayofyear'].values

for train_index, test_index in kf.split(X,y,grp):

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    model=RandomForestRegressor(n_estimators=150,random_state=42,max_features=8)

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    print(mean_absolute_percentage_error(y_test,preds))

    loss.append(mean_absolute_percentage_error(y_test,preds))

    s2models.append(model.predict(test_s2))
del s2models[2]
test.loc[test.segment==1, 'case_count']=np.mean(s1models,0)

test.loc[test.segment==2, 'case_count']=np.mean(s2models,0)
test.to_csv('submission.csv',index=False) 