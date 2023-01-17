# Loading necessary libraries

import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

import seaborn as sns

import time

from datetime import datetime
# Loading the data

data = pd.read_csv('../input/train.csv')
# data structure

data.head()
data.dtypes
# see data summary

data.describe()
# check for any missing values

data.isna().sum()
data.datetime = pd.to_datetime(data.datetime)

data['year'] = data.datetime.map(lambda x: x.strftime('%Y-%m-%d-%H-%M').split('-')[0])

data['month'] = data.datetime.map(lambda x: x.strftime('%Y-%m-%d-%H-%M').split('-')[1])

data['hour'] = data.datetime.map(lambda x: x.strftime('%Y-%m-%d-%H-%M').split('-')[3])

data['weekday'] = data.datetime.map(lambda x: x.weekday())
hour = data.drop(['datetime'],axis=1)

hour.rename(columns = {'count':'cnt'},inplace=True)
plt.figure(figsize=(15,15))

plt.subplots_adjust(hspace=0.5,wspace=0.2)



# season distribution

plt.subplot(4,3,1)

plt.bar(hour.season.unique(),hour.groupby('season').season.value_counts())

plt.xlabel('Season')



# holiday distribution

plt.subplot(4,3,2)

plt.bar(hour.holiday.unique(),hour.groupby('holiday').holiday.value_counts())

plt.xlabel('Holiday')



# weekday distribution

plt.subplot(4,3,3)

plt.bar(hour.weekday.unique(),hour.groupby('weekday').weekday.value_counts())

plt.xlabel('Weekday')



# workingday distribution

plt.subplot(4,3,4)

plt.bar(hour.workingday.unique(),hour.groupby('workingday').workingday.value_counts())

plt.xlabel('Workingday')



# weather distribution

plt.subplot(4,3,5)

plt.bar(hour.weather.unique(),hour.groupby('weather').weather.value_counts())

plt.xlabel('Weather')



# temp distribution

plt.subplot(4,3,6)

plt.hist(hour.temp)

plt.xlabel('Temp')



# temp distribution

plt.subplot(4,3,7)

plt.hist(hour.atemp)

plt.xlabel('aTemp')



# humidity distribution

plt.subplot(4,3,8)

plt.hist(hour.humidity)

plt.xlabel('Humidity')



# windspeed distribution

plt.subplot(4,3,9)

plt.hist(hour.windspeed)

plt.xlabel('Windspeed')



# year distribution

plt.subplot(4,3,10)

plt.bar(hour.year.unique(),hour.groupby('year').year.value_counts())

plt.xlabel('Year')



# month distribution

plt.subplot(4,3,11)

plt.bar(hour.month.unique(),hour.groupby('month').month.value_counts())

plt.xlabel('Month')



# hour distribution

plt.subplot(4,3,12)

plt.bar(hour.hour.unique(),hour.groupby('hour').hour.value_counts())

plt.xlabel('Hour')
# distribution of cnt with hr

plt.figure(figsize=(15,5))



# distribution of cnt with hr on working day

plt.subplot(1,2,1)

plt.bar(hour[hour.workingday == 1].hour.unique(),hour[hour.workingday == 1].groupby('hour').cnt.sum())

plt.xlabel('Hour')

plt.ylabel('Count')

plt.title('Working day')



# distribution of cnt with hr on non-working day

plt.subplot(1,2,2)

plt.bar(hour[hour.workingday == 0].hour.unique(),hour[hour.workingday == 0].groupby('hour').cnt.sum())

plt.xlabel('Hour')

plt.ylabel('Count')

plt.title('Non Working day')
plt.figure(figsize=(15,10))

plt.subplots_adjust(hspace=0.3,wspace=0.3)



# distrbution of cnt by weekday

plt.subplot(2,3,1)

plt.bar(hour.weekday.unique(),hour.groupby('weekday').cnt.sum())

plt.xlabel('Weekday')

plt.ylabel('Count')



# distrbution of cnt by workingday

plt.subplot(2,3,2)

plt.bar(hour.workingday.unique(),hour.groupby('workingday').cnt.sum())

plt.xlabel('Workingday')

plt.ylabel('Count')



# distrbution of cnt by Month

plt.subplot(2,3,3)

plt.bar(hour.month.unique(),hour.groupby('month').cnt.sum())

plt.xlabel('Month')

plt.ylabel('Count')



# distrbution of cnt by year

plt.subplot(2,3,4)

plt.bar(hour.year.unique(),hour.groupby('year').cnt.sum())

plt.xlabel('Year')

plt.ylabel('Count')



# distrbution of cnt by season

plt.subplot(2,3,5)

plt.bar(hour.season.unique(),hour.groupby('season').cnt.sum())

plt.xlabel('Season')

plt.ylabel('Count')



# distrbution of cnt by weather

plt.subplot(2,3,6)

plt.bar(hour.weather.unique(),hour.groupby('weather').cnt.sum())

plt.xlabel('Weather')

plt.ylabel('Count')
# distribution of casual and registered users separately with weekday

plt.figure(figsize=(15,15))

plt.subplots_adjust(hspace=0.3,wspace=0.2)



# distribution of registered users with weekday

plt.subplot(5,2,1)

plt.bar(hour.weekday.unique(),hour.groupby('weekday').registered.sum())

plt.xlabel('Weekday')

plt.ylabel('Count')

plt.title('Registered')



# distribution of casual users with weekday

plt.subplot(5,2,2)

plt.bar(hour.weekday.unique(),hour.groupby('weekday').casual.sum())

plt.xlabel('Weekday')

plt.ylabel('Count')

plt.title('Casual')



# distribution of registered users with working day

plt.subplot(5,2,3)

plt.bar(hour.workingday.unique(),hour.groupby('workingday').registered.sum())

plt.xlabel('Workingday')

plt.ylabel('Count')



# distribution of casual users with working day

plt.subplot(5,2,4)

plt.bar(hour.workingday.unique(),hour.groupby('workingday').casual.sum())

plt.xlabel('Workingday')

plt.ylabel('Count')



# distribution of registered users with month

plt.subplot(5,2,5)

plt.bar(hour.month.unique(),hour.groupby('month').registered.sum())

plt.xlabel('Month')

plt.ylabel('Count')



# distribution of casual users with month

plt.subplot(5,2,6)

plt.bar(hour.month.unique(),hour.groupby('month').casual.sum())

plt.xlabel('Month')

plt.ylabel('Count')



# distribution of registered users with hour

plt.subplot(5,2,7)

plt.bar(hour.hour.unique(),hour.groupby('hour').registered.sum())

plt.xlabel('Hour')

plt.ylabel('Count')



# distribution of casual users with hour

plt.subplot(5,2,8)

plt.bar(hour.hour.unique(),hour.groupby('hour').casual.sum())

plt.xlabel('Hour')

plt.ylabel('Count')



# distribution of registered users with season

plt.subplot(5,2,9)

plt.bar(hour.season.unique(),hour.groupby('season').registered.sum())

plt.xlabel('Season')

plt.ylabel('Count')



# distribution of casual users with season

plt.subplot(5,2,10)

plt.bar(hour.season.unique(),hour.groupby('season').casual.sum())

plt.xlabel('Season')

plt.ylabel('Count')
plt.figure(figsize=(15,10))



# distribution of cnt with temp

plt.subplot(2,2,1)

plt.scatter(hour.temp,hour.cnt)

plt.xlabel('Temp')

plt.ylabel('Count')



# distribution of cnt with atemp

plt.subplot(2,2,2)

plt.scatter(hour.atemp,hour.cnt)

plt.xlabel('aTemp')

plt.ylabel('Count')



# distribution of cnt with humidity

plt.subplot(2,2,3)

plt.scatter(hour.humidity,hour.cnt)

plt.xlabel('Humidity')

plt.ylabel('Count')



# distribution of cnt with windspeed

plt.subplot(2,2,4)

plt.scatter(hour.windspeed,hour.cnt)

plt.xlabel('Windspeed')

plt.ylabel('Count')
plt.figure(figsize=(15,7))

cor = hour.corr()

sns.heatmap(data=cor,annot=True)
hour['season'] = hour['season'].astype('category')

hour['holiday'] = hour['holiday'].astype('category')

hour['workingday'] = hour['workingday'].astype('category')

hour['weather'] = hour['weather'].astype('category')



hour['year'] = hour['year'].astype('int')

hour['month'] = hour['month'].astype('int')

hour['hour'] = hour['hour'].astype('int')

hour['weekday'] = hour['weekday'].astype('int')
hour.dtypes
def neg_rmlse(pred,actual):

    return -np.sqrt((np.sum((np.log(pred+1) - np.log(actual+1))**2))/len(pred))



# reason why rmlse has been defined with negative value will be explained in hyperparameter optimisation section below
# import the necessary library

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
# using only data left after removing 30 days data for our data modelling

x = hour.copy()

y = x.cnt



# dropping unnecessary independent variables and cnt from feature set x. Since atemp and temp are highly correlated,

# we are just gonna keep atemp in our model.

x.drop(['casual','registered','cnt','temp'],axis = 1,inplace=True)

x.dtypes
x_1 = x.copy()



# reinforce data types of year,month,hour,weekday

x_1['year'] = x_1['year'].astype('int')

x_1['month'] = x_1['month'].astype('int')

x_1['hour'] = x_1['hour'].astype('int')

x_1['weekday'] = x_1['weekday'].astype('int')



# splitting data intro train and test sets

x_train_1,x_test_1,y_train_1,y_test_1 = train_test_split(x_1,y,test_size=0.2,random_state=0)



# Training RF regressor model

model_1 = RandomForestRegressor(random_state = 0)

model_1.fit(x_train_1,y_train_1)



# predicting on train and test sets

pred_train_1 = model_1.predict(x_train_1)

pred_test_1 = model_1.predict(x_test_1)



# calculating errors on train and test sets

print ('Training error : ',-neg_rmlse(pred_train_1,y_train_1))

print ('Test error : ',-neg_rmlse(pred_test_1,y_test_1))



# feature importances

imp_1 = model_1.feature_importances_

indices_1 = np.argsort(imp_1)[::-1]

print ('\nFeature importances : ')

for f in range(x_1.shape[1]):

    print ("%d. %s (%f)" % (f + 1, x_1.columns[indices_1[f]], imp_1[indices_1[f]]))
x_2 = x.copy()



# redefine data types of year,month,hour,weekday

x_2['year'] = x_2['year'].astype('category')

x_2['month'] = x_2['month'].astype('category')

x_2['hour'] = x_2['hour'].astype('category')

x_2['weekday'] = x_2['weekday'].astype('category')



# splitting data intro train and test sets

x_train_2,x_test_2,y_train_2,y_test_2 = train_test_split(x_2,y,test_size=0.2,random_state=0)



# Training RF regressor model

model_2 = RandomForestRegressor(random_state = 0)

model_2.fit(x_train_2,y_train_2)



# predicting on train and test sets

pred_train_2 = model_2.predict(x_train_2)

pred_test_2 = model_2.predict(x_test_2)



# calculating errors on train and test sets

print ('Training error : ',-neg_rmlse(pred_train_2,y_train_2))

print ('Test error : ',-neg_rmlse(pred_test_2,y_test_2))



# feature importances

imp_2 = model_2.feature_importances_

indices_2 = np.argsort(imp_2)[::-1]

print ('\nFeature importances : ')

for f in range(x_2.shape[1]):

    print("%d. %s (%f)" % (f + 1, x_2.columns[indices_2[f]], imp_2[indices_2[f]]))
x_3 = x.copy()



# convert hour,month,weekday into cyclic variables by projecting them onto cos sin space and year into catergorical variable

x_3['hr_sin'] = np.sin(2.*np.pi*x_3.hour/24.)

x_3['hr_cos'] = np.cos(2.*np.pi*x_3.hour/24.)

x_3.drop(['hour'],axis=1,inplace=True)



x_3['mnth_sin'] = np.sin(2.*np.pi*x_3.month/24.)

x_3['mnth_cos'] = np.cos(2.*np.pi*x_3.month/24.)

x_3.drop(['month'],axis=1,inplace=True)



x_3['wd_sin'] = np.sin(2.*np.pi*x_3.weekday/24.)

x_3['wd_cos'] = np.cos(2.*np.pi*x_3.weekday/24.)

x_3.drop(['weekday'],axis=1,inplace=True)



x_3['year'] = x_3['year'].astype('category')



# splitting data intro train and test sets

x_train_3,x_test_3,y_train_3,y_test_3 = train_test_split(x_3,y,test_size=0.2,random_state=0)



# Training RF regressor model

model_3 = RandomForestRegressor(random_state = 0)

model_3.fit(x_train_3,y_train_3)



# predicting on train and test sets

pred_train_3 = model_3.predict(x_train_3)

pred_test_3 = model_3.predict(x_test_3)



# calculating errors on train and test sets

print ('Training error : ',-neg_rmlse(pred_train_3,y_train_3))

print ('Test error : ',-neg_rmlse(pred_test_3,y_test_3))



# feature importances

imp_3 = model_3.feature_importances_

indices_3 = np.argsort(imp_3)[::-1]

print ('\nFeature importances : ')

for f in range(x_3.shape[1]):

    print("%d. %s (%f)" % (f + 1, x_3.columns[indices_3[f]], imp_3[indices_3[f]]))
x_4 = x.copy()



# convert hr and mnth into cyclic variables by projecting them onto cos sin space

x_4['hr_sin'] = np.sin(2.*np.pi*x_4.hour/24.)

x_4['hr_cos'] = np.cos(2.*np.pi*x_4.hour/24.)

x_4.drop(['hour'],axis=1,inplace=True)



x_4['mnth_sin'] = np.sin(2.*np.pi*x_4.month/24.)

x_4['mnth_cos'] = np.cos(2.*np.pi*x_4.month/24.)

x_4.drop(['month'],axis=1,inplace=True)



x_4['wd_sin'] = np.sin(2.*np.pi*x_4.weekday/24.)

x_4['wd_cos'] = np.cos(2.*np.pi*x_4.weekday/24.)

x_4.drop(['weekday'],axis=1,inplace=True)



x_4['year'] = x_4['year'].astype('category')



x_4.drop(['windspeed'],axis=1,inplace=True)



# splitting data intro train and test sets

x_train_4,x_test_4,y_train_4,y_test_4 = train_test_split(x_4,y,test_size=0.2,random_state=0)



# Training RF regressor model

model_4 = RandomForestRegressor(random_state = 0)

model_4.fit(x_train_4,y_train_4)



# predicting on train and test sets

pred_train_4 = model_4.predict(x_train_4)

pred_test_4 = model_4.predict(x_test_4)



# calculating errors on train and test sets

print ('Training error : ',-neg_rmlse(pred_train_4,y_train_4))

print ('Test error : ',-neg_rmlse(pred_test_4,y_test_4))



# feature importances

imp_4 = model_4.feature_importances_

indices_4 = np.argsort(imp_4)[::-1]

print ('\nFeature importances : ')

for f in range(x_4.shape[1]):

    print("%d. %s (%f)" % (f + 1, x_4.columns[indices_4[f]], imp_4[indices_4[f]]))
x_5 = x.copy()



# convert hr and mnth into cyclic variables by projecting them onto cos sin space

x_5['hr_sin'] = np.sin(2.*np.pi*x_5.hour/24.)

x_5['hr_cos'] = np.cos(2.*np.pi*x_5.hour/24.)

x_5.drop(['hour'],axis=1,inplace=True)



x_5['mnth_sin'] = np.sin(2.*np.pi*x_5.month/24.)

x_5['mnth_cos'] = np.cos(2.*np.pi*x_5.month/24.)

x_5.drop(['month'],axis=1,inplace=True)



x_5['wd_sin'] = np.sin(2.*np.pi*x_5.weekday/24.)

x_5['wd_cos'] = np.cos(2.*np.pi*x_5.weekday/24.)

x_5.drop(['weekday'],axis=1,inplace=True)



x_5['year'] = x_5['year'].astype('category')



x_5.drop(['windspeed'],axis=1,inplace=True)



# target variables for casual and registered cnt

y_cas = hour.casual

y_reg = hour.registered



# splitting data intro train and test sets

x_train_5_cas,x_test_5_cas,y_train_5_cas,y_test_5_cas = train_test_split(x_5,y_cas,test_size=0.2,random_state=0)

x_train_5_reg,x_test_5_reg,y_train_5_reg,y_test_5_reg = train_test_split(x_5,y_reg,test_size=0.2,random_state=0)



# Training RF regressor model for casual cnt

model_5_cas = RandomForestRegressor(random_state = 0)

model_5_cas.fit(x_train_5_cas,y_train_5_cas)



# Training RF regressor model for casual cnt

model_5_reg = RandomForestRegressor(random_state = 0)

model_5_reg.fit(x_train_5_reg,y_train_5_reg)



# predicting total cnt by adding casual and registered cnt on train and test sets

pred_train_5 = model_5_cas.predict(x_train_5_cas) + model_5_reg.predict(x_train_5_reg)

pred_test_5 = model_5_cas.predict(x_test_5_cas) + model_5_reg.predict(x_test_5_reg)





# calculating errors on train and test sets

print ('Training error : ',-neg_rmlse(pred_train_5,y_train_5_cas+y_train_5_reg))

print ('Test error : ',-neg_rmlse(pred_test_5,y_test_5_cas+y_test_5_reg))
# import necessary libraries

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
# define the scoring function that will be used to get optimised hyperparameters



cv_scorer = make_scorer(neg_rmlse)             # neg_rmlse has been defined above



# The reason why neg remlse has been defined with negative return value is that the grid search cv always try to 

# maximise the scoring function.
# choose hyperparameter value space

estimators = [int(x) for x in np.linspace(50,500,num = 10)]

depths = [int(x) for x in np.linspace(10, 100, num = 10)]

depths.append(None)

features = ['auto','sqrt','log2']



# define parameter grid

param_grid2 = {"n_estimators": estimators,

               "max_depth" : depths,

               "max_features" : features}



# parameter tuning for model predicting casual cnt

model_cas_tune = RandomForestRegressor(random_state=0)

grid_search_cas = GridSearchCV(model_cas_tune,param_grid=param_grid2,scoring = cv_scorer,cv=3)

grid_search_cas.fit(x_train_5_cas, y_train_5_cas)



# parameter tuning for model predicting registered cnt

model_reg_tune = RandomForestRegressor(random_state=0)

grid_search_reg = GridSearchCV(model_reg_tune,param_grid=param_grid2,scoring = cv_scorer,cv=3)

grid_search_reg.fit(x_train_5_reg, y_train_5_reg)
print ('Best params for model predicting casual cnt : ',grid_search_cas.best_params_)

print ('Best score for model predicting casual cnt : ',-grid_search_cas.best_score_)

print ('\nBest params for model predicting registered cnt : ',grid_search_reg.best_params_)

print ('Best score for model predicting registered cnt : ',-grid_search_reg.best_score_)
x_5 = x.copy()



# convert hr and mnth into cyclic variables by projecting them onto cos sin space

x_5['hr_sin'] = np.sin(2.*np.pi*x_5.hour/24.)

x_5['hr_cos'] = np.cos(2.*np.pi*x_5.hour/24.)

x_5.drop(['hour'],axis=1,inplace=True)



x_5['mnth_sin'] = np.sin(2.*np.pi*x_5.month/24.)

x_5['mnth_cos'] = np.cos(2.*np.pi*x_5.month/24.)

x_5.drop(['month'],axis=1,inplace=True)



x_5['wd_sin'] = np.sin(2.*np.pi*x_5.weekday/24.)

x_5['wd_cos'] = np.cos(2.*np.pi*x_5.weekday/24.)

x_5.drop(['weekday'],axis=1,inplace=True)



x_5['year'] = x_5['year'].astype('category')



x_5.drop(['windspeed'],axis=1,inplace=True)



# target variables for casual and registered cnt

y_cas = hour.casual

y_reg = hour.registered



# splitting data intro train and test sets

x_train_5_cas,x_test_5_cas,y_train_5_cas,y_test_5_cas = train_test_split(x_5,y_cas,test_size=0.2,random_state=0)

x_train_5_reg,x_test_5_reg,y_train_5_reg,y_test_5_reg = train_test_split(x_5,y_reg,test_size=0.2,random_state=0)



# Training RF regressor model for casual cnt

model_cas_best = RandomForestRegressor(n_estimators = grid_search_cas.best_params_['n_estimators'],

                                       max_depth = grid_search_cas.best_params_['max_depth'],

                                       max_features = grid_search_cas.best_params_['max_features'],random_state = 0)

model_cas_best.fit(x_train_5_cas,y_train_5_cas)



# Training RF regressor model for casual cnt

model_reg_best = RandomForestRegressor(n_estimators = grid_search_reg.best_params_['n_estimators'],

                                       max_depth = grid_search_reg.best_params_['max_depth'],

                                       max_features = grid_search_reg.best_params_['max_features'],random_state = 0)

model_reg_best.fit(x_train_5_reg,y_train_5_reg)



# predicting total cnt by adding casual and registered cnt on train and test sets

pred_train_5_best = model_cas_best.predict(x_train_5_cas) + model_reg_best.predict(x_train_5_reg)

pred_test_5_best = model_cas_best.predict(x_test_5_cas) + model_reg_best.predict(x_test_5_reg)





# calculating errors on train and test sets

print ('Training error with best parameters : ',-neg_rmlse(pred_train_5_best,y_train_5_cas+y_train_5_reg))

print ('Test error with best parameters : ',-neg_rmlse(pred_test_5_best,y_test_5_cas+y_test_5_reg))