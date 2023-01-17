# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from dateutil.parser import parse



df=pd.read_csv('../input/train_users_2.csv')

#change into timestamp

df['timestamp_first_active']=pd.to_datetime(df['timestamp_first_active'],format='%Y%m%d%H%M%S')



print(df.info())

print(df.head())

print(df.describe())
# groupby year and counts values

df1=df.groupby(df['timestamp_first_active'].map(lambda x:x.year))['country_destination'].value_counts()

# make it into a dataframe

df2=df1.unstack()

#drop year 2009 data because it provides tiny info

df2=df2.drop(df2.index[0])



df2=df2.T

df2['Overall']=df2.sum(axis=1)

df2=df2.apply(lambda x: x/float(x.sum()))

df2.sort_values('Overall',ascending=False, inplace=True)

df2=df2.applymap(lambda x: "{:,.2%}".format(x))

pd.DataFrame(df2)
df1=df.groupby(df['timestamp_first_active'].map(lambda x:x.year))['first_device_type'].value_counts()

df5=df1.unstack()

df5=df5.drop(df5.index[0])

df5=df5.T

df5['Overall']=df5.sum(axis=1)



df5=df5.apply(lambda x: x/float(x.sum()))

df5.sort_values('Overall',ascending=False, inplace=True)

df5=df5.applymap(lambda x: "{:,.2%}".format(x))

pd.DataFrame(df5)
import matplotlib.pyplot as plt

%matplotlib inline

df3=df[['timestamp_first_active','id']]

df3=df3.set_index('timestamp_first_active')

#Resample by month

df3.resample('M').count().cumsum().plot(label='Cumulative trend')
print(df.age.isnull().sum()/df.age.sum())

df[df.age>1000].age.hist(bins=5)
va=df.age.values

df['age']=np.where(np.logical_and(df.age>1919,df.age<1995), 2015-df.age,df.age)

df['age']=np.where(np.logical_or(df.age>100,df.age<14),np.nan,df.age)

df['age'].hist(bins=10)
df_train=pd.read_csv('../input/train_users_2.csv')

df_text=pd.read_csv('../input/test_users.csv')

df_all=pd.concat((df_train,df_text), axis=0, ignore_index=True)

df_all['date_account_created']=pd.to_datetime(df_all['date_account_created'],format='%Y-%m-%d')

df_all['timestamp_first_active']=pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')

df_all.drop('date_first_booking', axis=1,inplace=True)

#va=df_all.age.values

df_all['age']=np.where(np.logical_or(df_all['age']>=90, df_all['age']<=15), np.nan, df_all.age)

df_all['age'].fillna(-1, inplace=True)

df_all['first_affiliate_tracked'].fillna(-1, inplace=True)

columns=['affiliate_channel','affiliate_provider','first_affiliate_tracked','first_browser','first_device_type','gender','signup_app','signup_method','language','signup_flow']

for column in columns:

    dummies=pd.get_dummies(df_all[column], prefix=column)

    df_all.drop(column,axis=1,inplace=True)

    df_all=pd.concat([df_all,dummies],axis=1)

df_all.head()
df_all['day_account_created']=df_all.date_account_created.dt.weekday

df_all['month_account_created']=df_all.date_account_created.dt.month

df_all['quater_account_created']=df_all.date_account_created.dt.quarter

df_all['year_account_created']=df_all.date_account_created.dt.year

df_all['Hour_first_active']=df_all.timestamp_first_active.dt.hour

df_all['day_first_active']=df_all.timestamp_first_active.dt.weekday

df_all['month_first_active']=df_all.timestamp_first_active.dt.month

df_all['quarter_first_active']=df_all.timestamp_first_active.dt.quarter

df_all['year_first_active']=df_all.timestamp_first_active.dt.year

df_all['created_less_active']=(df_all.date_account_created-df_all.timestamp_first_active).dt.days

columns_drop=['date_account_created','timestamp_first_active','country_destination']

for cl in columns_drop:

    if cl in df_all.columns:

        df_all.drop(cl, axis=1, inplace=True)

df_all.head()
sessions=pd.read_csv('../input/sessions.csv')

sessions_device=sessions[['user_id','device_type','secs_elapsed']]

secs=sessions_device.groupby(['user_id','device_type'], as_index=False, sort=False).sum()

idx=secs.groupby(['user_id'],sort=False)['secs_elapsed'].transform(max)==secs['secs_elapsed']

df_f=pd.DataFrame(secs[idx])

df_f.drop('device_type',axis=1,inplace=True)

df_f.head()
remain=secs.drop(secs.index[idx])

idx1=remain.groupby(['user_id'],sort=False)['secs_elapsed'].transform(max)==remain['secs_elapsed']

df_s=pd.DataFrame(remain[idx1])

df_s.drop('device_type',axis=1,inplace=True)

df_s.head()
action=sessions[['user_id','action']].fillna('not provided')

action.loc[:,'count']=1

action1=action.groupby(['user_id','action'],as_index=False, sort=False).sum()

action2=action1.pivot(index='user_id', columns='action',values='count')

at1=action2.fillna(0)



action_type=sessions[['user_id','action_type']].fillna('not provided')

action_type.loc[:,'count']=1

action_type1=action_type.groupby(['user_id','action_type'],as_index=False, sort=False).sum()

at2=action_type1.pivot(index='user_id', columns='action_type',values='count')

at2=at2.fillna(0)



at3=sessions[['user_id','action_detail']].fillna('not provided')

at3.loc[:,'count']=1

at3=at3.groupby(['user_id','action_detail'],as_index=False, sort=False).sum()

at3=at3.pivot(index='user_id', columns='action_detail',values='count')

at3=at3.fillna(0)



at3.head()
action_data=pd.concat([at1,at2,at3],axis=1,join='inner')

df_f.rename(columns={'secs_elapsed': 'secs_elapsed_high'}, inplace=True)

df_s.rename(columns={'secs_elapsed': 'secs_elapsed_low'}, inplace=True)

device_data=pd.merge(df_f,df_s, on='user_id',how='outer')

device_data.set_index('user_id',inplace=True)

device_data.head()

da_data=pd.concat([action_data,device_data],axis=1,join='outer')

da_data.fillna(0, inplace=True)

df_all.set_index('id',inplace=True)

df_all=pd.concat([df_all,da_data],axis=1,join='outer')

df_all.head()
df_train.set_index('id', inplace=True)

df_train=pd.concat([df_train, df_all], axis=1, join='inner')
len(df_train)
from sklearn.preprocessing import LabelEncoder





df_train = pd.concat([df_train['country_destination'], df_all], axis=1, join='inner')



id_train = df_train.index.values

labels = df_train['country_destination']

le = LabelEncoder()

y = le.fit_transform(labels)

X = df_train.drop('country_destination', axis=1, inplace=False)
from sklearn import cross_validation,decomposition,grid_search

import xgboost as xgb



XGB_model=xgb.XGBClassifier(objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)

#param_grid={'max_depth':[3,4,5],'learning_rate':[0.1,0.3],'n_estimators':[25,50]}

#Due to the limited running time restriction, I chose to run the best parameter worked out by my own computer

#param_grid={'max_depth':4,'learning_rate':0.3,'n_estimators':50}

model=grid_search.GridSearchCV(estimator=XGB_model, param_grid=param_grid, scoring='accuracy', verbose=10, n_jobs=1,iid=True, refit=True, cv=3)
model.fit(X, y)

print("Best score: %0.3f" % model.best_score_)

print("Best parameters set:")

best_parameters = model.best_estimator_.get_params()

for param_name in sorted(param_grid.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))

    

# Prepare test data for prediction

df_test.set_index('id', inplace=True)

df_test = pd.merge(df_test.loc[:,['date_first_booking']], df_all, how='left', left_index=True, right_index=True, sort=False)

X_test = df_test.drop('date_first_booking', axis=1, inplace=False)

X_test = X_test.fillna(-1)

id_test = df_test.index.values



# Make predictions

y_pred = model.predict_proba(X_test)

# Taking the 5 classes with highest probabilities

ids = []  #list of ids

cts = []  #list of countries

for i in range(len(id_test)):

    idx = id_test[i]

    ids += [idx] * 5

    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()



#Generate submission

print("Outputting final results...")

sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])

sub.to_csv('./submission.csv',index=False)


