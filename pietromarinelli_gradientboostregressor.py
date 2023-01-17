import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import poisson
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

sample_submission = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')
test = test.rename(columns = {'ForecastId' : 'Id'})
train = train.drop(columns = ['County' , 'Province_State'])

test = test.drop(columns = ['County' , 'Province_State'])
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = train.iloc[:,1].values

train.iloc[:,1] = labelencoder.fit_transform(X.astype(str))



X = train.iloc[:,5].values

train.iloc[:,5] = labelencoder.fit_transform(X)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

X = test.iloc[:,1].values

test.iloc[:,1] = labelencoder.fit_transform(X)



X = test.iloc[:,5].values

test.iloc[:,5] = labelencoder.fit_transform(X)
train.Date = pd.to_datetime(train.Date).dt.strftime("%Y%m%d").astype(int)

test.Date = pd.to_datetime(test.Date).dt.strftime("%Y%m%d").astype(int)

test.head()
x = train.iloc[:,1:6]

y = train.iloc[:,6]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test =train_test_split(x,y, test_size = 0.2, random_state = 0 )
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
X_test = test.iloc[:,1:6]

predictor = pipeline_dt.predict(X_test)
prediction_list = [x for x in predictor]
sub = pd.DataFrame({'Id': test.index , 'TargetValue': prediction_list})
sub['TargetValue'].value_counts()
p = sub.copy()

q = sub.copy()

r = sub.copy()
p['TargetValue']=p[['TargetValue']].applymap(lambda x: poisson.ppf(0.05, x)) #.quantile(q=0.05).reset_index()

q['TargetValue']=q[['TargetValue']]#.applymap(lambda x: poisson.ppf(0.95, x))#.quantile(q=0.5).reset_index()

r['TargetValue']=r[['TargetValue']].applymap(lambda x: poisson.ppf(0.95, x))#.quantile(q=0.95).reset_index()
p.loc[p['TargetValue'].isnull(),'TargetValue'] = p['TargetValue'].mean()

r.loc[r['TargetValue'].isnull(),'TargetValue'] = r['TargetValue'].mean()
p.columns = ['Id' , 'q0.05']

q.columns = ['Id' , 'q0.5']

r.columns = ['Id' , 'q0.95']
p = pd.concat([p,q['q0.5'] , r['q0.95']],1)
#p['q0.05']=p['q0.05'].clip(0,10000)

#p['q0.05']=p['q0.5'].clip(0,10000)

#p['q0.05']=p['q0.95'].clip(0,10000)

#p
p['Id'] =p['Id']+ 1

p
sub=pd.melt(p, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.loc[sub['TargetValue']<0,'TargetValue'] = 0

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub
#sub['TargetValue'].describe()