# Importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sn

from scipy import stats

from numpy import median

from sklearn import metrics

from sklearn.metrics import mean_squared_log_error, r2_score,mean_squared_error

from sklearn.model_selection import train_test_split

import catboost
df_train = pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')

df_train.head(5)
df_train.shape
#Check for null values



df_train.isnull().sum()
train_date = pd.DatetimeIndex(df_train['datetime'])

df_train['year'] = train_date.year

df_train['month'] = train_date.month

df_train['hour'] = train_date.hour

df_train['weekday'] = train_date.dayofweek

df_train.head()
fig, axes = plt.subplots(nrows=2,ncols=2)

fig.set_size_inches(12, 10)

sn.boxplot(data=df_train,y="count",orient="v",ax=axes[0][0])

sn.boxplot(data=df_train,y="count",x="season",orient="v",ax=axes[0][1])

sn.boxplot(data=df_train,y="count",x="hour",orient="v",ax=axes[1][0])

sn.boxplot(data=df_train,y="count",x="workingday",orient="v",ax=axes[1][1])



axes[0][0].set(ylabel='Count',title="Box Plot On Count")

axes[0][1].set(xlabel='Season', ylabel='Count',title="Box Plot On Count Across Season")

axes[1][0].set(xlabel='Hour Of The Day', ylabel='Count',title="Box Plot On Count Across Hour Of The Day")

axes[1][1].set(xlabel='Working Day', ylabel='Count',title="Box Plot On Count Across Working Day")
fig, ((ax1,ax9),(ax2,ax10),(ax3,ax11),(ax4,ax12),(ax5,ax13),(ax6,ax14),(ax7,ax15),(ax8,ax16)) = plt.subplots(8, 2,figsize=(15,25))

fig.suptitle('Casual vs Registered')

sn.barplot(x = df_train['season'], y = df_train['casual'],ax = ax1)

sn.barplot(x = df_train['year'], y = df_train['casual'],ax = ax2)

sn.barplot(x = df_train['month'], y = df_train['casual'],ax = ax3)

sn.barplot(x = df_train['hour'], y = df_train['casual'],ax = ax4)

sn.barplot(x = df_train['holiday'], y = df_train['casual'],ax = ax5)

sn.barplot(x = df_train['weekday'], y = df_train['casual'],ax = ax6)

sn.barplot(x = df_train['workingday'], y = df_train['casual'],ax = ax7)

sn.barplot(x = df_train['weather'], y = df_train['casual'],ax = ax8)

sn.barplot(x = df_train['season'], y = df_train['registered'],ax = ax9)

sn.barplot(x = df_train['year'], y = df_train['registered'],ax = ax10)

sn.barplot(x = df_train['month'], y = df_train['registered'],ax = ax11)

sn.barplot(x = df_train['hour'], y = df_train['registered'],ax = ax12)

sn.barplot(x = df_train['holiday'], y = df_train['registered'],ax = ax13)

sn.barplot(x = df_train['weekday'], y = df_train['registered'],ax = ax14)

sn.barplot(x = df_train['workingday'], y = df_train['registered'],ax = ax15)

sn.barplot(x = df_train['weather'], y = df_train['registered'],ax = ax16)
# Regression plot is used to verify if a pattern can be observed between `count` and numerical variables



fig,[[ax1,ax4],[ax2,ax5],[ax3,ax6]] = plt.subplots(3,2, figsize = (15,15))

plt.rc('xtick', labelsize=10) 

plt.rc('ytick', labelsize=10) 



sn.regplot(x = 'temp', y = 'casual',data = df_train, ax = ax1)

ax1.set(title="Relation between temperature and casual")

sn.regplot(x = 'humidity', y = 'casual',data = df_train, ax = ax2)

ax2.set(title="Relation between humidity and casual")

sn.regplot(x = 'windspeed', y = 'casual',data = df_train, ax = ax3)

ax3.set(title="Relation between windspeed and casual")

sn.regplot(x = 'temp', y = 'registered',data = df_train, ax = ax4)

ax4.set(title="Relation between temperature and registered")

sn.regplot(x = 'humidity', y = 'registered',data = df_train, ax = ax5)

ax5.set(title="Relation between humidity and registered")

sn.regplot(x = 'windspeed', y = 'registered',data = df_train, ax = ax6)

ax6.set(title="Relation between windspeed and registered")
data_corr = df_train[['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered', 'count']].corr()

mask = np.array(data_corr)

mask[np.tril_indices_from(mask)] = False

fig = plt.subplots(figsize=(15, 10))

sn.heatmap(data_corr, mask=mask, vmax=1, square=True, annot=True, cmap="YlGnBu")
fig,axes = plt.subplots(ncols=2,nrows=2)

fig.set_size_inches(12, 10)

sn.distplot((df_train["count"]),ax=axes[0][0])

stats.probplot((df_train["count"]), dist='norm', fit=True, plot=axes[0][1])

sn.distplot(np.log(df_train["count"]),ax=axes[1][0])

stats.probplot(np.log1p(df_train["count"]), dist='norm', fit=True, plot=axes[1][1])
df_train_withoutoutliers = df_train[np.abs(df_train["count"]-df_train["count"].mean())<=(3.5*df_train["count"].std())]
print ("Shape Of The Before Ouliers: ",df_train.shape)

print ("Shape Of The After Ouliers: ",df_train_withoutoutliers.shape)
# to write it easier 

df=df_train_withoutoutliers
df.drop(['datetime','atemp','windspeed'],inplace=True,axis=1) 
# SPLITIING

X = df[[col for col in df.columns if col not in ['casual','registered', 'count']]]

y = df[['casual','registered']]



X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Log to make values "normal"

yLog1 = np.log1p(y_train['casual'])

yLog2 = np.log1p(y_train['registered'])
columns = list(X_train.columns)

cat_features = np.where(X_train[columns].dtypes != np.float)[0]

print(cat_features)
from catboost import CatBoostRegressor

model=CatBoostRegressor(iterations=4000, depth=10,one_hot_max_size=1, learning_rate=0.01, loss_function='RMSE',colsample_bylevel=0.7

                        ,l2_leaf_reg=1,silent=True,cat_features=cat_features)

model.fit(X_train,yLog1)
from catboost import CatBoostRegressor

model2=CatBoostRegressor(iterations=4000, depth=10,one_hot_max_size=1, learning_rate=0.01, loss_function='RMSE',colsample_bylevel=0.7

                         ,l2_leaf_reg=1,silent=True,cat_features=cat_features)

model2.fit(X_train,yLog2)
casual=model.predict(X_test)

registered=model2.predict(X_test)

preds=np.expm1(casual)+np.expm1(registered)

Y=y_test['registered']+y_test['casual']

print('RMSLE:',(np.sqrt(mean_squared_log_error(preds,Y))))

print('R2:', r2_score(preds,Y))
ax = sn.regplot(Y, preds, x_bins = 200)

ax.set(title = "Comparison between the actual vs predicted values")