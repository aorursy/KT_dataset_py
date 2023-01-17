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

import matplotlib.pyplot as plt
import seaborn as sns
        
from scipy import stats        
# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/bike-sharing-demand/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/bike-sharing-demand/test.csv')
test.head()
train_two=train.drop(['casual', 'registered'], axis=1)
train_two.head()
test_two=test
test_two.head()
train_two.isnull().sum()
train_two.info()
test_two.info()
train_two['datetime']=train_two['datetime'].astype('datetime64')
train_two.info()
train['datetime']=train['datetime'].astype('datetime64')
train.info()
test_two['datetime']=test_two['datetime'].astype('datetime64')
test_two.info()
#train_two.set_index('datetime', inplace=True)
sns.distplot(train_two['count'], hist=True)
plt.show()
stats.probplot(train_two['count'], plot=plt)
plt.show()
stats.shapiro(train_two['count'])
train_two[['temp', 'atemp', 'humidity', 'windspeed']].head()
sns.distplot(train_two['temp'], hist=True)
plt.show()
stats.probplot(train_two['temp'], plot=plt)
plt.show()
sns.distplot(train_two['humidity'], hist=True)
plt.show()
stats.probplot(train_two['humidity'], plot=plt)
plt.show()
sns.distplot(train_two['windspeed'], hist=True)
plt.show()
stats.probplot(train_two['windspeed'], plot=plt)
plt.show()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scale_stand=scaler.fit_transform(train_two[['temp', 'atemp', 'humidity', 'windspeed']])
print('스케일 조정 전 features Min value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].min(axis=0)))
print('스케일 조정 전 features Max value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].max(axis=0)))
print('스케일 조정 후 features Min value : \n {}'.format(X_train_scale_stand.min(axis=0)))
print('스케일 조정 후 features Max value : \n {}'.format(X_train_scale_stand.max(axis=0)))
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
X_train_scale_robust=scaler.fit_transform(train_two[['temp', 'atemp', 'humidity', 'windspeed']])
print('스케일 조정 전 features Min value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].min(axis=0)))
print('스케일 조정 전 features Max value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].max(axis=0)))
print('스케일 조정 후 featcures Min value : \n {}'.format(X_train_scale_robust.min(axis=0)))
print('스케일 조정 후 features Max value : \n {}'.format(X_train_scale_robust.max(axis=0)))
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train_scale_minmax=scaler.fit_transform(train_two[['temp', 'atemp', 'humidity', 'windspeed']])
print('스케일 조정 전 features Min value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].min(axis=0)))
print('스케일 조정 전 features Max value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].max(axis=0)))
print('스케일 조정 후 features Min value : \n {}'.format(X_train_scale_minmax.min(axis=0)))
print('스케일 조정 후 features Max value : \n {}'.format(X_train_scale_minmax.max(axis=0)))
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
X_train_scale_normal=scaler.fit_transform(train_two[['temp', 'atemp', 'humidity', 'windspeed']])
print('스케일 조정 전 features Min value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].min(axis=0)))
print('스케일 조정 전 features Max value : \n {}'.format(train_two[['temp', 'atemp', 'humidity', 'windspeed']].max(axis=0)))
print('스케일 조정 후 features Min value : \n {}'.format(X_train_scale_normal.min(axis=0)))
print('스케일 조정 후 features Max value : \n {}'.format(X_train_scale_normal.max(axis=0)))
#from sklearn.svm import SVC
#svc=SVC()
#svc.fit(train_two[train_two.columns.difference(['count'])], train_two['count'])
train_two[train_two.columns.difference(['season', 'weather', 'workingday'])].corr()
train_two.head()
train_two.info()
train_two.astype({'season' : 'category', 'holiday' : 'category', 'workingday' : 'category', 'weather' : 'category'}).dtypes
train_two[['season_spring', 'season_summer', 'season_fall', 'season_winter']]=pd.get_dummies(train_two['season'], prefix=['season'])
train_two.drop('season', axis=1, inplace=True)
train_two.head()
train_two[['holiday_true', 'holiday_false']]=pd.get_dummies(train_two['holiday'], prefix=['holiday'])
train_two.drop('holiday', axis=1, inplace=True)
train_two.head()
train_two[['workingday_true', 'workingday_false']]=pd.get_dummies(train_two['workingday'], prefix=['workingday'])
train_two.drop('workingday', axis=1, inplace=True)
train_two.head()
train_two[['weather_1', 'weather_2', 'weather_3', 'weather_4']]=pd.get_dummies(train_two['weather'], prefix=['weather'])
train_two.drop('weather', axis=1, inplace=True)
train_two.head()
sns.distplot(train_two['count'], hist=True)
plt.show()
train_two['year']=train['datetime'].dt.year
train_two['month']=train['datetime'].dt.month
train_two['weekday']=train['datetime'].dt.weekday
train_two['hour']=train['datetime'].dt.hour
train_two.head()
test_two['year']=test['datetime'].dt.year
test_two['month']=test['datetime'].dt.month
test_two['weekday']=test['datetime'].dt.weekday
test_two['hour']=test['datetime'].dt.hour
test.head()
categoryVariableList=['year', 'month', 'weekday', 'hour']
for var in categoryVariableList:
    train_two[var]=train_two[var].astype('category')
train_two.info()    
train_two.drop(['datetime'], axis=1, inplace=True)
train_two.head()
test_two.head()
categoryVariableList=['year', 'month', 'weekday', 'hour', 'season', 'weather', 'holiday', 'workingday']
for var in categoryVariableList:
    test_two[var]=test_two[var].astype('category')
test_two.info()    
#train_two.dtypes.value_counts()
#dataTypeDf = pd.DataFrame(train_two.dtypes.value_counts()).reset_index().rename(columns={"index":"variableType",0:"count"})
#dataTypeDf
#fig,ax = plt.subplots()
#fig.set_size_inches(12,5)
#sns.barplot(data=train_two,x="variableType",y="count",ax=ax)
#ax.set(xlabel='variableTypeariable Type', ylabel='Count',title="Variables DataType Count")
#이거 왜 에러 해결 못하겟지? 
fig, axes=plt.subplots(nrows=4, ncols=3)
fig.set_size_inches(12, 10)
sns.boxplot(data=train_two, y='count', orient='v', ax=axes[0][0])
sns.boxplot(data=train_two, y='count', x='season', orient='v', ax=axes[0][1])
sns.boxplot(data=train_two, y='count', x='holiday', orient='v', ax=axes[0][2])
sns.boxplot(data=train_two, y='count', x='holiday', orient='v', ax=axes[1][0])
sns.boxplot(data=train_two, y='count', x='weather', orient='v', ax=axes[1][1])
sns.boxplot(data=train_two, y='count', x='year', orient='v', ax=axes[1][2])
sns.boxplot(data=train_two, y='count', x='month', orient='v', ax=axes[2][0])
sns.boxplot(data=train_two, y='count', x='weekday', orient='v', ax=axes[2][1])
sns.boxplot(data=train_two, y='count', x='hour', orient='v', ax=axes[2][2])
sns.boxplot(data=train_two, y='count', x='workingday', orient='v', ax=axes[3][0])
train_two_nonOutliers=train_two[np.abs(train_two['count']-train_two['count'].mean())<=(3*train_two['count'].std())]
print('Shape of The before Outliers : ', train_two.shape)
print('Shape of The after Outliers : ', train_two_nonOutliers.shape)
train_two.info()
corrMatt=train_two[['temp', 'atemp', 'humidity', 'count']].corr()
mask=np.array(corrMatt)
np.array(corrMatt)
mask[np.tril_indices_from(mask)]=False
fig, ax=plt.subplots()
fig.set_size_inches(20, 10)
sns.heatmap(corrMatt, mask=mask, vmax=.8, square=True, annot=True)
fig, (ax1, ax2, ax3)=plt.subplots(ncols=3)
fig.set_size_inches(12, 5)
sns.regplot(x='temp', y='count', data=train_two, ax=ax1)
sns.regplot(x='atemp', y='count', data=train_two, ax=ax2)
sns.regplot(x='humidity', y='count', data=train_two, ax=ax3)
fig, axes=plt.subplots(ncols=2, nrows=2)
fig.set_size_inches(12, 10)
sns.distplot(train_two['count'], ax=axes[0][0])
stats.probplot(train_two['count'], dist='norm', fit=True, plot=axes[0][1])
sns.distplot(np.log(train_two_nonOutliers["count"]),ax=axes[1][0])
stats.probplot(np.log1p(train_two_nonOutliers["count"]), dist='norm', fit=True, plot=axes[1][1])
fig, (ax1, ax2, ax3, ax4)  = plt.subplots(nrows=4)
fig.set_size_inches(12, 20)
sortOrder = ['January','February','March','April','May','June','July','August','September','October','November','December']
hueOrder = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']

monthAggregated=pd.DataFrame(train_two.groupby('month')['count'].mean()).reset_index()
monthSorted=monthAggregated.sort_values(by='count', ascending=False)
sns.barplot(data=monthSorted, x='month', y='count', ax=ax1)
ax1.set(xlabel='Month', ylabel='Average Count', title='Average Count By Month')

hourAggregated=pd.DataFrame(train_two.groupby(['hour', 'season'], sort=True)['count'].mean()).reset_index()
sns.pointplot(x=hourAggregated['hour'], y=hourAggregated['count'], hue=hourAggregated['season'], data=hourAggregated, join=True, ax=ax2)
ax2.set(xlabel='Hour Of The Day', ylabel='Users Count', title='Average Users Count By Hour Of  The Day Across Season', label='big')

hourAggregated=pd.DataFrame(train_two.groupby(['hour', 'weekday'], sort=True)['count'].mean()).reset_index()
sns.pointplot(x=hourAggregated['hour'], y=hourAggregated['count'], hue=hourAggregated['weekday'], data=hourAggregated, join=True, ax=ax3)
ax3.set(xlabel='Hour Of The Day', ylabel='Users Count', title='Average Users Count By Hour Of The Day Across Weekdays', label='big')

hourAggregated=pd.DataFrame(train_two.groupby(['hour', 'month'], sort=True)['count'].mean()).reset_index()
sns.pointplot(x=hourAggregated['hour'], y=hourAggregated['count'], hue=hourAggregated['month'], data=hourAggregated, join=True, ax=ax4)
ax4.set(xlabel='Hour Of The Day', ylabel='Users Count', title='Average Users Count By Hour Of The Day Across Months', label='big')
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logistic regression model
lModel = LinearRegression()

yLabels=train_two['count']

# Train the model
yLabelsLog = np.log1p(yLabels)
lModel.fit(X = train_two.drop(['count'], axis=1), y = yLabelsLog)


# Make predictions
preds = lModel.predict(X= test_two)
preds=pd.DataFrame(preds)
submission=pd.read_csv('/kaggle/input/bike-sharing-demand/sampleSubmission.csv')
submission.head()
submission['count']=preds
submission.head()
submission.to_csv('Linear Regression Mode.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=100)
yLabelsLog = np.log1p(yLabels)
rfModel.fit(train_two.drop(['count'], axis=1), yLabelsLog)
preds = rfModel.predict(X= test_two)
preds=pd.DataFrame(preds)
preds.head()
submission.head()
submission['count']=preds
submission.head()
submission.to_csv('Ensemble Models - Random Forest.csv', index=False)


