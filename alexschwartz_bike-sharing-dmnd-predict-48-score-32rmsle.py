#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

sns.set(style="dark")

sns.set(style="whitegrid", color_codes=True)
train=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')

print('train shape:',train.shape)

print('test shape:',test.shape)
train.head()
test.head()
#check for null data

train.isnull().sum()
import missingno as msno



fig,ax=plt.subplots(2,1,figsize=(10,5))



msno.matrix(train,ax=ax[0])

ax[0].set_title('Train Data')

msno.matrix(test,ax=ax[1])

ax[1].set_title('Test Data')
#variable datatype:

train.info()
from datetime import datetime

from dateutil import parser

import calendar



#parse string datetime into datetime format

train['datetime2']=train.datetime.apply(lambda x: parser.parse(x))



#Get some different time variables

train['year']=train.datetime2.apply(lambda x: x.year)

train['month']=train.datetime2.apply(lambda x: x.month)

train['weekday']=train.datetime2.apply(lambda x: x.weekday())

train['weekday_name']=train.datetime2.apply(lambda x: calendar.day_name[x.weekday()])

train['hour']=train.datetime2.apply(lambda x: x.hour)

#create categorical data

train['season_decode']=train.season.map({1:'spring',2:'summer',3:'fall',4:'winter'})

train['working_decode']=train.workingday.map({1:'work',0:'notwork'})

train['weather_decode']=train.weather.map({1:'Clear',2:'Mist',3:'LightRain',4:'HeavyRain'})
train.head()
f,ax=plt.subplots(1,2)

sns.distplot(train['count'],bins=30,ax=ax[0])

ax[0].set_title('count distrib')

sns.boxplot(data=train,y=train['count'],ax=ax[1])

ax[1].set_title('count boxplot')
mean_count=train['count'].mean()

std_count=train['count'].std()

print(mean_count-3*std_count)

print(mean_count+3*std_count)

outliers1=train[train['count']>(mean_count+3*std_count)]

len(outliers1['count'])
train2=train[train['count']<=(mean_count+3*std_count)]

train2.shape
#Season

sns.boxplot(data=train2,y=train2['count'],x=train['season_decode']).set_title('Demand by season')
#Year



train2.groupby(['year','month'])['count'].mean().plot().set_title('demand by year')

#WeekDay & Hour:

week_hour=train2.groupby(['weekday_name','hour'])['count'].mean().unstack()

week_hour=week_hour.reindex(index=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])





plt.figure(figsize=(15,6))

cmap2 = sns.cubehelix_palette(start=2,light=1, as_cmap=True)



sns.heatmap(week_hour,cmap=cmap2).set_title('Demand by Day-Hour')
#Difference between casual and resgitered

train2.groupby(['hour'])['casual','registered','count'].mean().plot().set_title('Demand by hour')



train2.groupby(['weekday_name'])['casual','registered','count'].mean().plot(kind='bar').set_title('demand by day of week')

#Weather

train2.groupby(['weather_decode'])['casual','registered'].mean().plot(kind='bar').set_title('demand by weather')
#Temp

season_temp=train2.groupby(['season_decode','temp'])['count'].mean().unstack()





plt.figure(figsize=(15,8))

cmap3 = sns.cubehelix_palette(start=6,light=1, as_cmap=True)



sns.heatmap(season_temp,cmap=cmap3).set_title('demand by season and temperature')
Correlation_Matrix=train2[['holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered','count']].corr()

mask = np.array(Correlation_Matrix)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(Correlation_Matrix,mask=mask,vmax=.8,annot=True,square=True)
#preparing data sets for random forest

X=train2[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','year','month','weekday','hour']]



y_count=train2['count']

y_casual=train2['casual']

y_reg=train2['registered']
from sklearn.preprocessing import StandardScaler



#Scaled all distributions

X_Scaled=StandardScaler().fit_transform(X=X)
from sklearn.model_selection import train_test_split

#Split for train-test

X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y_count, test_size=0.25, random_state=42)

from sklearn.ensemble import RandomForestRegressor



rf_count=RandomForestRegressor()

rf_count.fit(X_train,y_train)



importance_count=pd.DataFrame(rf_count.feature_importances_ , index=X.columns, columns=['count']).sort_values(by='count',ascending=False)



importance_count.plot(kind='bar',color='r').set_title('Importance of features for total demand')
#repeat for casual demand:



X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y_casual, test_size=0.25, random_state=42)



rf_casual=RandomForestRegressor()

rf_casual.fit(X_train,y_train)



importance_casual=pd.DataFrame(rf_casual.feature_importances_ , index=X.columns, columns=['casual']).sort_values(by='casual',ascending=False)

importance_casual.plot(kind='bar').set_title('Importance of features for casual demand')
#repeat for registered demand:



X_train, X_test, y_train, y_test = train_test_split(X_Scaled, y_reg, test_size=0.25, random_state=42)



rf_reg=RandomForestRegressor()

rf_reg.fit(X_train,y_train)



importance_reg=pd.DataFrame(rf_reg.feature_importances_ , index=X.columns, columns=['reg']).sort_values(by='reg',ascending=False)

importance_reg.plot(kind='bar',color='g').set_title('Importance of features for registered demand')
importance_df=pd.concat([importance_count,importance_casual,importance_reg],axis=1)

importance_df.plot(kind='bar').set_title('Feature importance for each kind of demand')
feature_selection=['workingday','weather','atemp','humidity','windspeed','year','month','weekday','hour']

print('features for model:',len(feature_selection))
#Prepare Training data

X_train=train2[feature_selection]

print(X_train.shape)



y_train=train2['count']

print(y_train.shape)
#Prepare Test data



#parse string datetime into datetime format

test['datetime2']=test.datetime.apply(lambda x: parser.parse(x))



#Get some different time variables

test['year']=test.datetime2.apply(lambda x: x.year)

test['month']=test.datetime2.apply(lambda x: x.month)

test['weekday']=test.datetime2.apply(lambda x: x.weekday())

test['hour']=test.datetime2.apply(lambda x: x.hour)



X_test=test[feature_selection]

print(X_test.shape)
X_train_scaled=StandardScaler().fit_transform(X=X_train)

X_test_scaled=StandardScaler().fit_transform(X=X_test)
from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import make_scorer





def rmsle(y,y_pred):

    return np.sqrt(mean_squared_log_error(y,y_pred))

    

rmsle_score=make_scorer(rmsle)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



rfr=RandomForestRegressor(random_state=42)



score=cross_val_score(rfr,X_train_scaled,y_train,cv=15,scoring=rmsle_score)



print(f'Score rmsle mean: {np.round(score.mean(),4)}')

print(f'Score  rmsle std: {np.round(score.std(),4)}')
rfr.fit(X_train_scaled,y_train)

y_pred=rfr.predict(X_test_scaled)
submission=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

submission['count']=y_pred

submission.to_csv('submissionI.csv',index=False)
#Without Scaling Data



rfr.fit(X_train,y_train)

y_pred=rfr.predict(X_test)

submission2=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

submission2['count']=y_pred

submission2.to_csv('submissionII.csv',index=False)               
from sklearn.model_selection import GridSearchCV, train_test_split





x_train2,x_test2,y_train2,y_test2=train_test_split(X_train,y_train,test_size=0.25,random_state=42)



params={'n_estimators': [10,50,100,300,500],

       'n_jobs':[-1],

       'max_features':['auto','sqrt','log2'],

       'random_state':[42]}



rfr_tuned=GridSearchCV(estimator=RandomForestRegressor(),param_grid=params,scoring='neg_mean_squared_log_error',verbose=True)



rfr_tuned.fit(x_train2,y_train2)

print(rfr_tuned.best_params_)

print(rfr_tuned.best_estimator_)



from sklearn.ensemble import RandomForestRegressor



rfr_final=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,

                      oob_score=False, random_state=42, verbose=0,

                      warm_start=False)



rfr_final.fit(x_train2,y_train2)

y_pred2=rfr_final.predict(x_test2)

print('RMSLE:',np.round(rmsle(y_test2,y_pred2),4))
rfr_final=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,

                      max_features='auto', max_leaf_nodes=None,

                      min_impurity_decrease=0.0, min_impurity_split=None,

                      min_samples_leaf=1, min_samples_split=2,

                      min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=-1,

                      oob_score=False, random_state=42, verbose=0,

                      warm_start=False)



rfr_final.fit(X_train,y_train)

y_pred=rfr.predict(X_test)

submission3=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')

submission3['count']=y_pred

submission3.to_csv('submissionIII.csv',index=False)