import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')

df=pd.read_csv('../input/weather-covid19/weather.csv')
df.head()
test.loc[test['Province_State'].isnull(),'Province_State']='unknown'

test['country_region']=test['Country_Region']+"_"+test['Province_State']

test.head()
train.loc[train['Province_State'].isnull(),'Province_State']='unknown'

train['country_region']=train['Country_Region']+"_"+train['Province_State']

train.head()
group=df.groupby('country_region')['temp_max_mean_week'].mean()



group=group.reset_index()



group.rename(columns={'temp_max_mean_week':'temp_max'})



train=pd.merge(train,group,on=['country_region'],how='left')

test=pd.merge(test,group,on=['country_region'],how='left')
group=df.groupby('country_region')['temp_min_mean_week'].mean()



group=group.reset_index()



group.rename(columns={'temp_min_mean_week':'temp_min'})

train=pd.merge(train,group,on=['country_region'],how='left')

test=pd.merge(test,group,on=['country_region'],how='left')
group=df.groupby('country_region')['mean_week_humidity'].mean()



group=group.reset_index()



group.rename(columns={'mean_week_humidity':'temp_min'})

train=pd.merge(train,group,on=['country_region'],how='left')

test=pd.merge(test,group,on=['country_region'],how='left')
train.head()
test.head()
test = test.rename(columns = {'ForecastId' : 'Id'})

train = train.drop(columns = ['County' , 'Province_State','Country_Region'])

test = test.drop(columns = ['County' , 'Province_State','Country_Region'])
train.fillna(-999,inplace=True)
test.fillna(-999,inplace=True)
train.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = train.iloc[:,6].values

train.iloc[:,6] = labelencoder.fit_transform(X.astype(str))



X = train.iloc[:,4].values

train.iloc[:,4] = labelencoder.fit_transform(X)
train.head()
test.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = test.iloc[:,5].values

test.iloc[:,5] = labelencoder.fit_transform(X)



X = test.iloc[:,4].values

test.iloc[:,4] = labelencoder.fit_transform(X)
test.head()
train['Date'] = pd.to_datetime(train['Date'])

test['Date'] = pd.to_datetime(test['Date'])



train['day']=train['Date'].dt.dayofyear

test['day']=test['Date'].dt.dayofyear

train.head()



train.head()
# Mean encoding of the train set

train['country_mean']=train.groupby(['Population','Target'])['TargetValue'].transform('mean')
# mean encoding of the test set



group=train.groupby(['Population','Target'],as_index=False)['TargetValue'].mean()

group.rename(columns={'TargetValue':'country_mean'},inplace=True)

test=pd.merge(test,group,on=['Population','Target'],how='left')

test.head()
# let us see by adding the lag features

print(train.shape)

print(test.shape)
train.fillna(0,inplace=True)

train.head()
#x = train.iloc[:,1:7]



x=train.drop(columns=['TargetValue'],axis=1)

y=train['TargetValue']

x.pop('Id')

x.pop('Date')

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2, random_state = 0 )
train.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

pipeline_dt = Pipeline([('scaler2' , StandardScaler()),

                        ('RandomForestRegressor: ', RandomForestRegressor())])

pipeline_dt.fit(x_train , y_train)

prediction = pipeline_dt.predict(x_test)
score = pipeline_dt.score(x_test,y_test)

print('Score: ' + str(score))
from sklearn import metrics

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(prediction,y_test)

print(val_mae)
test.head()
# #x_test.shape

test.pop('Date')

X_test = test.iloc[:,1:10]

predictor = pipeline_dt.predict(X_test)

# X_test.shape

# test.shape



# print(test.head())

# print(x_test.head())
prediction_list = [x for x in predictor]
sub = pd.DataFrame({'Id': test.index , 'TargetValue': prediction_list})
sub['TargetValue'].value_counts()
p=sub.groupby(['Id'])['TargetValue'].quantile(q=0.05).reset_index()

q=sub.groupby(['Id'])['TargetValue'].quantile(q=0.5).reset_index()

r=sub.groupby(['Id'])['TargetValue'].quantile(q=0.95).reset_index()
p.columns = ['Id' , 'q0.05']

q.columns = ['Id' , 'q0.5']

r.columns = ['Id' , 'q0.95']
p = pd.concat([p,q['q0.5'] , r['q0.95']],1)
p['q0.05']=p['q0.05'].clip(0,10000)

p['q0.05']=p['q0.5'].clip(0,10000)

p['q0.05']=p['q0.95'].clip(0,10000)

p
p['Id'] =p['Id']+ 1

p
sub=pd.melt(p, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub