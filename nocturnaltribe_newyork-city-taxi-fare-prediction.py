import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import geopy.distance as geo

import datetime
import time
import calendar

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from sklearn import preprocessing

from sklearn import metrics
%matplotlib inline
#df = pd.read_csv('../input/train.csv').sample(250000)
df =  pd.read_csv('../input/train.csv', nrows = 2000000)
df.head()
df.info()
unwanted_indices = df[ (abs(df['pickup_latitude']) > 90) | 
                       (abs(df['dropoff_latitude']) > 90) 
                     ].index
df.drop(list(unwanted_indices), inplace=True)

del unwanted_indices
unwanted_indices = df[ (abs(df['pickup_longitude']) > 180) | 
                       (abs(df['dropoff_longitude']) > 180) 
                     ].index
df.drop(list(unwanted_indices), inplace=True)
df[ (abs(df['pickup_longitude']) > 180) | 
                       (abs(df['dropoff_longitude']) > 180) 
                     ].index
df[ (abs(df['pickup_latitude']) > 90) | 
                       (abs(df['dropoff_latitude']) > 90) 
                     ].index
unwanted_indices = df[df['dropoff_latitude'].isnull()].index
df.drop(list(unwanted_indices), inplace=True)

unwanted_indices = df[df['dropoff_longitude'].isnull()].index
df.drop(list(unwanted_indices), inplace=True)
#
#      Passanger number
#
if 'unwanted_indices' in globals(): del unwanted_indices

unwanted_indices = df[ (df['passenger_count'] > 6) | (df['passenger_count'] == 0) ].index
df.drop(list(unwanted_indices), inplace=True)
df.head()
unwanted_indices = df[df['fare_amount']<=0].index
df.drop(list(unwanted_indices), inplace=True)
df.head(2)
#
#      distance
#      USA:  horizontal - 2,680 miles ; vertical : 1,582 miles.
#
# if 'unwanted_indices' in globals(): del unwanted_indices
# unwanted_indices = df[ df['Travel distance'] > 500 ].index
# df.drop(list(unwanted_indices), inplace=True)
df['Travel distance'] = list(  map( lambda x1,x2,x3,x4: 
                               geo.distance( (x3,x1), (x4,x2) ).miles,
                               df['pickup_longitude'], df['dropoff_longitude'],
                               df['pickup_latitude'],  df['dropoff_latitude'] ) )
newtimeStamp = pd.to_datetime( df['pickup_datetime'].apply( lambda x: x.split(' UTC')[0]) )
#pd.to_datetime(df['pickup_datetime'])
df['Hour']  = newtimeStamp.apply(lambda x : x.hour)
df['Month'] = newtimeStamp.apply(lambda x : x.month)
df['Date']  = newtimeStamp.apply(lambda x : x.date())
for x in df['pickup_datetime'] :
    y = x.split(' ')[2]
    if( y!='UTC' ) :
        print( y )
Day_of_Week=newtimeStamp.apply(
    lambda x : 
    calendar.day_name[datetime.date(x.year,x.month,x.day)
                      .weekday()]
)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
Day_of_Week = newtimeStamp.apply(
    lambda x : 
    dmap[datetime.date(x.year,x.month,x.day)
            .weekday()]
)

df['Day of Week'] = Day_of_Week
df['qty'] = df['pickup_datetime'].apply( lambda x: 1)
df['timestamp'] = newtimeStamp.apply(lambda x : time.mktime(
                             ( x.year,x.month,x.day,
                               x.hour,x.minute,x.second,
                               1,x.day,-1) )
                  )
df.head()
sns.countplot( x='passenger_count',data=df )
# plt.yscale('log')
# plt.ylim(1,10**5)
# plt.show()
sns.lmplot("Travel distance", "fare_amount", data=df, fit_reg=False, hue='passenger_count',aspect=1.2)
plt.xscale('log')
plt.xlim(0.01,10**4)
plt.yscale('log')
plt.ylim(1,10**3)
plt.show()
sns.kdeplot( df['fare_amount'] )
sns.kdeplot(df['Travel distance'])
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(0.5,100)
#plt.ylim(0.000001,1)
#plt.show()
sns.countplot(x='Day of Week', data=df)
sns.factorplot(x='Day of Week',data=df,hue='passenger_count',
               kind='count', log=True, size=5, aspect=1.8)
dfm=df.groupby(['Day of Week','Hour']).count()
fp=dfm['qty'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis')
dfm=df.groupby(['Day of Week','Month']).count()
fp=dfm['qty'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis')
dfm=df.groupby(['Day of Week','Hour']).mean()
fp=dfm['Travel distance'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
dfm=df.groupby(['Day of Week','Month']).mean()
fp=dfm['Travel distance'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis',robust=True,annot=True)
sns.distplot(df['Travel distance'], hist=True, kde=False, 
             bins=40, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
# plt.yscale('log')
# plt.ylim(10**-1,10**2)
# plt.show()
sns.kdeplot(df['Travel distance'])
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(10,1000)
# plt.ylim(0.0000001,0.0001)
# plt.show()
dfm=df.groupby('Date').sum()
dfm['qty'].plot(figsize=(10,5),grid=True,style='k.')
pd.plotting.lag_plot(dfm['qty'],alpha=0.5)
dfm=df.groupby('Date').mean()
pd.plotting.autocorrelation_plot(dfm['Travel distance'],alpha=0.75)
#plt.xlim(0,30)
plt.ylim(-0.2,0.2)
plt.show()
dfm=df.groupby('Month').count()
dfm['qty'].plot.line(figsize=(10,5),grid=True)
sns.lmplot("Travel distance", "fare_amount", data=df, fit_reg=False, hue='Hour',aspect=1.5)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.001,1000)
plt.ylim(1,300)
plt.show()
col=list(df.columns)
col
dftrain = df.drop([
 'key',
 'pickup_datetime',
 'Hour',
 'Month',
 'Date',
 'Day of Week',
 'qty'
                   ],axis=1).dropna()
dftrain.head(2)
dftrain.info()
X_train = dftrain.drop('fare_amount',axis=1)
y_train = dftrain['fare_amount']
XX_train, XX_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.30)
# import sklearn.pipeline
# from sklearn.linear_model import LinearRegression

# scaler = sklearn.preprocessing.StandardScaler()

# lm = LinearRegression(fit_intercept=True)
# steps = [('feature_selection', scaler),
#         ('regression', lm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# # fit pipeline on X_train and y_train
# pipeline.fit( XX_train, np.log(yy_train) )
# #pipeline.fit( XX_train, yy_train )

# # call pipeline.predict() on X_test data to make a set of test predictions
# yy_predictions = pipeline.predict( XX_test )
# predictions = np.exp( yy_predictions )
# #predictions = yy_predictions

# sns.distplot(yy_predictions,bins=1000)
# # plt.yscale('log')
# # plt.ylim(0.00001,1)
# plt.xlabel("Predicted Fare")
# plt.ylabel("Distribution of observations")
# # plt.show()
# for yy in predictions:
#     if yy<0 :
#         print(yy)
# coeff_df = pd.DataFrame(lm.coef_,X_train.columns,columns=['Coefficient'])
# coeff_df
# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))
# plt.scatter(yy_test,predictions)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,1000)
# plt.ylim(1,1000)
# plt.xlabel("Actual Fare")
# plt.ylabel("Predicted Fare")
# plt.show()
# import sklearn.pipeline
# from sklearn.linear_model import TheilSenRegressor

# scaler = sklearn.preprocessing.StandardScaler()

# lm = TheilSenRegressor()
# steps = [('feature_selection', scaler),
#         ('regression', lm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# # fit pipeline on X_train and y_train
# #pipeline.fit( XX_train, np.log(yy_train) )
# pipeline.fit( XX_train, yy_train )

# # call pipeline.predict() on X_test data to make a set of test predictions
# yy_predictions = pipeline.predict( XX_test )
# #predictions = np.exp( yy_predictions )
# predictions = yy_predictions

# for yy in predictions: 
#     if yy!=yy :
#         print(yy)
# plt.scatter(yy_test,predictions)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,1000)
# plt.ylim(1,1000)
# plt.xlabel("Actual Fare")
# plt.ylabel("Predicted Fare")
# plt.show()
# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))
# import sklearn.pipeline
# from sklearn import svm

# scaler = sklearn.preprocessing.StandardScaler()

# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'C': [50,75,100],#[0.1,1, 10, 100], 
#     'gamma': [40,20,1],#[1,0.1,0.01,0.001,0.0001], 
#     'kernel': ['rbf'],
#     'tol':[0.001] 
# } 
# grid = GridSearchCV( svm.SVR(), param_grid,refit=True, verbose=3, n_jobs=4 )


# steps = [('feature_selection', scaler),
#         ('regression', grid)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# # May take awhile!
# # fit pipeline on X_train and y_train
# #pipeline.fit( XX_train, np.log(yy_train) )
# pipeline.fit( XX_train, yy_train )

# # May take awhile!
# yy_predictions = pipeline.predict(XX_test)

# predictions = yy_predictions #np.exp( yy_predictions )

# print(grid.best_params_)
# print(grid.best_estimator_)

# for yy in predictions:
#     if yy<0 :
#         print(yy)
        
# sns.distplot(predictions,bins=1000)
# plt.xlabel("Predicted Fare")
# plt.ylabel("Distribution of observations")
# plt.show()
# plt.scatter(yy_test,predictions)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,1000)
# plt.ylim(1,1000)
# plt.xlabel("Actual Fare")
# plt.ylabel("Predicted Fare")
# plt.show()
# eps = (yy_test-predictions)
# # Density Plot and Histogram of travel distances
# sns.distplot(eps, hist=True, kde=True, 
#              bins=100, color = 'darkblue', 
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 2})
# plt.xlabel('error - difference between fares')
# plt.show()
# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))
# import sklearn.pipeline
# scaler = sklearn.preprocessing.StandardScaler()

# import tensorflow as tf
# feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

# from tensorflow.contrib import learn

# classifier = learn.DNNRegressor(
#     feature_columns=feature_columns,
#     hidden_units=[20,40,20],
#     optimizer=tf.train.ProximalAdagradOptimizer(
#       learning_rate=0.1,
#       l1_regularization_strength=0.01
#     ))

# steps = [('scaler', scaler),
#         ('DNNclassifier', classifier)]
# pipeline = sklearn.pipeline.Pipeline(steps)

# ### fit pipeline on X_train and y_train
# pipeline.fit( XX_train, yy_train, DNNclassifier__steps=800)

# ### call pipeline.predict() on X_test data to make a set of test predictions
# predictions = pipeline.predict( XX_test )
# predictions = list(pd.DataFrame(predictions)[0])

# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))
# plt.scatter(yy_test,predictions)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,100)
# plt.ylim(1,100)
# plt.xlabel("Actual Fare")
# plt.ylabel("Predicted Fare")
# plt.show()
# import sklearn.pipeline
# scaler = sklearn.preprocessing.StandardScaler()

# import lightgbm as lgb
# # train
# gbm = lgb.LGBMRegressor(objective='regression',
#                         boosting_type='gbdt',
#                         num_leaves=100,#1001,
#                         learning_rate=0.003,
#                         n_estimators=1000)

# steps = [('scaler', scaler),
#         ('GBM', gbm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# ### fit pipeline on X_train and y_train
# pipeline.fit( XX_train, yy_train)

# ### call pipeline.predict() on X_test data to make a set of test predictions
# predictions = pipeline.predict( XX_test )

# print('MAE:', metrics.mean_absolute_error(yy_test, predictions))
# print('MSE:', metrics.mean_squared_error(yy_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(yy_test, predictions)))

# plt.scatter(yy_test,predictions)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlim(1,100)
# plt.ylim(1,100)
# plt.xlabel("Actual Fare")
# plt.ylabel("Predicted Fare")
# plt.show()
dftest = pd.read_csv('../input/test.csv')
dftest.info()
dftest.head()
newtimeStamp=pd.to_datetime(dftest['pickup_datetime'])
dftest['Hour']  = newtimeStamp.apply(lambda x : x.hour)
dftest['Month'] = newtimeStamp.apply(lambda x : x.month)
dftest['Date']  = newtimeStamp.apply(lambda x : x.date())
#
# If other than UTC time zone
#
for x in dftest['pickup_datetime'] :
    y = x.split(' ')[2]
    if( y!='UTC' ) :
        print( y )
newtimeStamp[0]
newtimeStamp[0].date()
import datetime
import calendar
Day_of_Week=newtimeStamp.apply(
    lambda x : 
    calendar.day_name[datetime.date(x.year,x.month,x.day)
                      .weekday()]
)

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
Day_of_Week = newtimeStamp.apply(
    lambda x : 
    dmap[datetime.date(x.year,x.month,x.day)
            .weekday()]
)

dftest['Day of Week'] = Day_of_Week
dftest['qty'] = dftest['pickup_datetime'].apply( lambda x: 1)
newtimeStamp.head(2)
dftest['timestamp'] = newtimeStamp.apply(lambda x : time.mktime(
                              (x.year,x.month,x.day,
                               x.hour,x.minute,x.second,
                               1,x.day,-1) ))
import geopy.distance as geo

dftest['Travel distance'] = list(map( lambda x1,x2,x3,x4: 
                   geo.distance( (x3,x1), (x4,x2) ).miles,
                   dftest['pickup_longitude'], dftest['dropoff_longitude'],
                   dftest['pickup_latitude'],  dftest['dropoff_latitude'] ))
sns.countplot( x='passenger_count',data=dftest )
# plt.yscale('log')
# plt.ylim(100,10**4)
# plt.show()
sns.kdeplot(dftest['Travel distance'])
plt.xscale('log')
plt.yscale('log')
plt.xlim(10**-1,10**2)
plt.ylim(10**-5,1)
plt.show()
dftest.head()
sns.countplot(x='Day of Week', data=dftest)
sns.factorplot(x='Day of Week',data=dftest,hue='passenger_count',
               kind='count', log=True, size=5, aspect=1.8)
#
#sns.countplot(x='Day of Week',data=dftest,hue='passenger_count',palette='viridis')
#plt.legend(loc=2, bbox_to_anchor=(1.05,1),borderaxespad=0.)
# list(dfm.columns)
dfm=dftest.groupby(['Day of Week','Hour']).count()
fp=dfm['qty'].unstack()
fp
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis')
dfm=dftest.groupby(['Day of Week','Month']).count()
fp=dfm['qty'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
plt.figure(figsize=(12,6))
sns.clustermap(fp,cmap='viridis')
dfm=dftest.groupby(['Day of Week','Hour']).mean()
fp=dfm['Travel distance'].unstack()
#fp
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis')
dfm=dftest.groupby(['Day of Week','Month']).mean()
fp=dfm['Travel distance'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(fp,cmap='viridis',robust=True,annot=True)
dfm=dftest.groupby('Date').sum()
dfm['qty'].plot(figsize=(10,5),grid=True,style='k.')
plt.yscale('log')
plt.ylim(0.1,1000)
plt.show()
pd.plotting.lag_plot(dfm['qty'],alpha=0.5)
plt.xlim(0,30)
plt.ylim(0,30)
plt.show()
dfm=dftest.groupby('Date').mean()

pd.plotting.autocorrelation_plot(dfm['Travel distance'],alpha=0.75)
#plt.xlim(0,30)
plt.ylim(-0.12,0.12)
plt.show()
dfm=dftest.groupby('Month').count()
dfm['qty'].plot.line(figsize=(10,5),grid=True)
from sklearn import preprocessing
list(X_train.columns)
df.head(2)
dftest.head(2)
coltest=list(dftest.columns)
idy=coltest[2:7]
idy.append(coltest[12])
idy.append(coltest[13])
idy
X_train = dftrain[idy]
y_train = dftrain['fare_amount']

X_test  = dftest[idy]
len(y_train)
X_test.head(2)
X_train.head(2)
# X_test.head(2)
# X_train.head(2)
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report,confusion_matrix
# import sklearn.pipeline
# from sklearn.linear_model import TheilSenRegressor

# scaler = sklearn.preprocessing.StandardScaler()
# lm = TheilSenRegressor()
# steps = [('feature_selection', scaler),
#         ('regression', lm)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# # fit pipeline on X_train and y_train
# pipeline.fit( X_train, np.log(y_train) )

# # call pipeline.predict() on X_test data to make a set of test predictions
# y_predictions = pipeline.predict( X_test )
# predictions = np.exp( y_predictions )

# #print the intercept
# print(lm.intercept_)

# coeff_df = pd.DataFrame(lm.coef_,X_train_scaled.columns,columns=['Coefficient'])
# coeff_df
# import sklearn.pipeline
# from sklearn import svm

# scaler = sklearn.preprocessing.StandardScaler()

# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'C': [300,350,400],#[10,50,100,200,300,400], 
#     'gamma': [125,100,75],#[100,75,50,25,1],
#     'kernel': ['rbf'],
#     'tol':[0.001] 
# } 
# grid = GridSearchCV( svm.SVR(), param_grid,refit=True, verbose=3, n_jobs=8 )


# steps = [('feature_selection', scaler),
#         ('regression', grid)]

# pipeline = sklearn.pipeline.Pipeline(steps)

# # May take awhile!
# # fit pipeline on X_train and y_train
# #pipeline.fit( X_train, np.log(y_train) )
# pipeline.fit( X_train, y_train )

# # May take awhile!
# # y_grdsvc = pipeline.predict(X_test)
# # predictions = np.exp( y_grdsvc )
# predictions = pipeline.predict(X_test)

# print(grid.best_params_)
# print(grid.best_estimator_)
# import sklearn.pipeline
# scaler = sklearn.preprocessing.StandardScaler()

# import tensorflow as tf
# feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)

# from tensorflow.contrib import learn

# classifier = learn.DNNRegressor(
#     feature_columns=feature_columns,
#     hidden_units=[20,40,20],
#     optimizer=tf.train.ProximalAdagradOptimizer(
#       learning_rate=0.1,
#       l1_regularization_strength=0.01
#     ))

# steps = [('scaler', scaler),
#         ('DNNclassifier', classifier)]
# pipeline = sklearn.pipeline.Pipeline(steps)

# ### fit pipeline on X_train and y_train
# pipeline.fit( X_train, y_train, DNNclassifier__steps=1000)

# ### call pipeline.predict() on X_test data to make a set of test predictions
# predictions = pipeline.predict( X_test )
# predictions = list(pd.DataFrame(predictions)[0])
import sklearn.pipeline
scaler = sklearn.preprocessing.StandardScaler()

import lightgbm as lgb
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        boosting_type='gbdt',
                        num_leaves=1001,
                        learning_rate=0.01,
                        n_estimators=2500)

steps = [('scaler', scaler),
        ('GBM', gbm)]

pipeline = sklearn.pipeline.Pipeline(steps)

### fit pipeline on X_train and y_train
pipeline.fit( X_train, y_train)

### call pipeline.predict() on X_test data to make a set of test predictions
predictions = pipeline.predict( X_test )
for yy in predictions:
    if yy<0 :
        print(yy)
temp = pd.DataFrame(predictions,columns=['Predicted Fare'])
df_pred = dftest[['Travel distance','Hour']].join(temp)
#
#print( df_pred['Travel distance'].max(), df_pred['Predicted Fare'].max() )
#
sns.lmplot("Travel distance", "Predicted Fare", data=df_pred, 
           fit_reg=False, hue='Hour', scatter_kws={'alpha':0.5}, aspect=1.8)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.001,500)
plt.ylim(1,300)
plt.show()
sns.lmplot("Travel distance", "fare_amount", data=df, fit_reg=False, 
            hue='Hour', scatter_kws={'alpha':0.5}, aspect=1.8)
plt.xscale('log')
plt.yscale('log')
plt.xlim(0.001,500)
plt.ylim(1,300)
plt.show()
fare = []
for y_pred in predictions:
    fare.append( '{:.{prec}f}'.format(y_pred, prec=2) ) 
print(len(fare),len(predictions))
pd.DataFrame( { 'key':list(dftest['key']),
                'fare_amount':fare } ).set_index('key').to_csv('sample_submission.csv', sep=',')