import os

dir='../input'

os.chdir(dir)
import pandas as pd

from pandas import DataFrame, Series

import numpy as np
city=pd.read_csv('city_attributes.csv')

humidity=pd.read_csv('humidity.csv')

pressure=pd.read_csv('pressure.csv')

temp=pd.read_csv('humidity.csv')

descrip=pd.read_csv('weather_description.csv')

winddir=pd.read_csv('wind_direction.csv')

windspeed=pd.read_csv('wind_speed.csv')
humidity.shape
humidity.head()
humidity.describe()
#Combined dataset of city Vancouver:

vancouver=(humidity[['datetime','Vancouver']].merge(temp[['datetime','Vancouver']],on='datetime').merge(pressure[['datetime','Vancouver']], on='datetime').merge(windspeed[['datetime','Vancouver']], on='datetime').merge(winddir[['datetime','Vancouver']], on='datetime').merge(descrip[['datetime','Vancouver']], on='datetime'))
#Weather data of city Vancouver:

vancouver.columns=['datetime','humidity','temperature','pressure','windspeed','winddirection','description']

vancouver.head()
vancouver.shape
vancouver.describe()
vancouver.info()
import datetime

vancouver['datetime']=pd.to_datetime(vancouver['datetime'])

vancouver.info()
vancouver.isnull().sum()
vancouver['humidity']=vancouver['humidity'].fillna(method='bfill')

vancouver['temperature']=vancouver['temperature'].fillna(method='bfill')

vancouver['pressure']=vancouver['pressure'].fillna(method='bfill')

vancouver['windspeed']=vancouver['windspeed'].fillna(method='bfill')

vancouver['winddirection']=vancouver['winddirection'].fillna(method='bfill')

vancouver['description']=vancouver['description'].fillna(method='bfill')
vancouver.isnull().sum()
vancouver['humidity']=vancouver['humidity'].fillna(method='ffill')

vancouver['temperature']=vancouver['temperature'].fillna(method='ffill')

vancouver['pressure']=vancouver['pressure'].fillna(method='ffill')

vancouver['windspeed']=vancouver['windspeed'].fillna(method='ffill')

vancouver['winddirection']=vancouver['winddirection'].fillna(method='ffill')

vancouver['description']=vancouver['description'].fillna(method='ffill')
vancouver.isnull().sum()
vancouver.head()
vancouver.describe()
vancouver['datetime'].head()
time=pd.DatetimeIndex(vancouver['datetime'])

vancouver['date']=time.date

vancouver['year']=time.year

vancouver['month']=time.month

vancouver['day']=time.day

vancouver['time']=time.time
vancouver.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Visualization to check the distribution of the attributes:

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

sns.distplot(vancouver['pressure'],color='blue')

plt.title('Distribution of pressure')

plt.subplot(3,3,2)

sns.distplot(vancouver['temperature'],color='orange')

plt.title('Distribution of temperature')

plt.subplot(3,3,3)

sns.distplot(vancouver['humidity'],color='magenta')

plt.title('Distribution of humidity')

plt.subplot(3,3,4)

sns.distplot(vancouver['windspeed'],color='brown')

plt.title('Distribution of wind speed')

plt.subplot(3,3,5)

sns.distplot(vancouver['winddirection'],color='yellow')

plt.title('Distribution of wind direction')

plt.tight_layout()

plt.show()
#Distribution of values across each factors:

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

plt.scatter(x=vancouver['pressure'],y=vancouver['temperature'],color='magenta')

plt.title('Pressure vs Temperature')

plt.xlabel('Pressure')

plt.ylabel('Temperature')

plt.subplot(3,3,2)

plt.scatter(x=vancouver['humidity'],y=vancouver['pressure'],color='orange')

plt.title('Humidity vs Pressure')

plt.xlabel('Humdity')

plt.ylabel('Pressure')

plt.subplot(3,3,3)

plt.scatter(x=vancouver['temperature'],y=vancouver['humidity'],color='green')

plt.title('Temperature vs Humidity')

plt.xlabel('Temperature')

plt.ylabel('Humidity')

plt.subplot(3,3,4)

plt.scatter(x=vancouver['windspeed'],y=vancouver['winddirection'],color='blue')

plt.title('Wind speed vsWin direction')

plt.xlabel('Wind speed')

plt.ylabel('Wind direction')

plt.tight_layout()

plt.show()
#Boxplot of each attributes across year:

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

sns.boxplot(x='year',y='temperature',data=vancouver)

plt.title('Temperature across year box plot')

plt.subplot(3,3,2)

sns.boxplot(x='year',y='pressure',data=vancouver)

plt.title('Pressure across year box plot')

plt.subplot(3,3,3)

sns.boxplot(x='year',y='humidity',data=vancouver)

plt.title('Humidity across year box plot')

plt.subplot(3,3,4)

sns.boxplot(x='year',y='windspeed',data=vancouver)

plt.title('Wind speed across year box plot')

plt.subplot(3,3,5)

sns.boxplot(x='year',y='winddirection',data=vancouver)

plt.title('Wind direction across year box plot')

plt.tight_layout()

plt.show()
#Boxplot of each attributes across month:

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

sns.boxplot(x='month',y='temperature',data=vancouver)

plt.title('Temperature across month box plot')

plt.subplot(3,3,2)

sns.boxplot(x='month',y='pressure',data=vancouver)

plt.title('Pressure across month box plot')

plt.subplot(3,3,3)

sns.boxplot(x='month',y='humidity',data=vancouver)

plt.title('Humidity across month box plot')

plt.subplot(3,3,4)

sns.boxplot(x='month',y='windspeed',data=vancouver)

plt.title('Wind speed across month box plot')

plt.subplot(3,3,5)

sns.boxplot(x='month',y='winddirection',data=vancouver)

plt.title('Wind direction across month box plot')

plt.tight_layout()

plt.show()
#Visualization of climatic attributes across each year:

from matplotlib import style

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

vancouver.groupby('year')['pressure'].mean().plot(kind='bar')

plt.title("Average pressure across each year")

plt.subplot(3,3,2)

vancouver.groupby('year')['temperature'].mean().plot(kind='bar')

plt.title("Average temperature across each year")

plt.subplot(3,3,3)

vancouver.groupby('year')['humidity'].mean().plot(kind='bar')

plt.title("Average humidity across each year")

plt.subplot(3,3,4)

vancouver.groupby('year')['windspeed'].mean().plot(kind='bar')

plt.title("Average wind speed across each year")

plt.subplot(3,3,5)

vancouver.groupby('year')['winddirection'].mean().plot(kind='bar')

plt.title("Average wind direction across each year")

plt.tight_layout()

style.use('classic')

plt.show()
#Visualization of climatic attributes across each month:

from matplotlib import style

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

vancouver.groupby('month')['pressure'].mean().plot(kind='bar')

plt.title("Average pressure across each month")

plt.subplot(3,3,2)

vancouver.groupby('month')['temperature'].mean().plot(kind='bar')

plt.title("Average temperature across each month")

plt.subplot(3,3,3)

vancouver.groupby('month')['humidity'].mean().plot(kind='bar')

plt.title("Average humidity across each month")

plt.subplot(3,3,4)

vancouver.groupby('month')['windspeed'].mean().plot(kind='bar')

plt.title("Average wind speed across each month")

plt.subplot(3,3,5)

vancouver.groupby('month')['winddirection'].mean().plot(kind='bar')

plt.title("Average wind direction across each month")

plt.tight_layout()

style.use('classic')

plt.show()
#Visualization of weather description acros each climatic attributes:

plt.figure(figsize=(30,30))

plt.subplot(5,1,1)

vancouver.groupby('description').count()['pressure'].plot(kind='bar',color='yellow')

plt.title("Weather description across pressure")

plt.subplot(5,1,2)

vancouver.groupby('description').count()['temperature'].plot(kind='bar',color='orange')

plt.title("Weather description across temperature")

plt.subplot(5,1,3)

vancouver.groupby('description').count()['humidity'].plot(kind='bar',color='brown')

plt.title("Weather description across humidity")

plt.subplot(5,1,4)

vancouver.groupby('description').count()['windspeed'].plot(kind='bar',color='magenta')

plt.title("Weather description across windspeed")

plt.subplot(5,1,5)

vancouver.groupby('description').count()['winddirection'].plot(kind='bar',color='blue')

plt.title("Weather description across winddirection")

plt.tight_layout()
#Correlation across each factors:

vancouver.corr()
#Heat map to visualize correlation:

sns.heatmap(vancouver.corr(),annot=True)
data=pd.DataFrame()

data['avghumidity']=vancouver.groupby('date')['humidity'].mean()

data['maxhumidity']=vancouver.groupby('date')['humidity'].max()

data['minhumidity']=vancouver.groupby('date')['humidity'].min()

data['avgtemp']=vancouver.groupby('date')['temperature'].mean()

data['maxtemp']=vancouver.groupby('date')['temperature'].max()

data['mintemp']=vancouver.groupby('date')['temperature'].min()

data['avgpressure']=vancouver.groupby('date')['pressure'].mean()

data['maxpressure']=vancouver.groupby('date')['pressure'].max()

data['minpressure']=vancouver.groupby('date')['pressure'].min()

data['avgwindspeed']=vancouver.groupby('date')['windspeed'].mean()

data['maxwindspeed']=vancouver.groupby('date')['windspeed'].max()

data['minwindspeed']=vancouver.groupby('date')['windspeed'].min()

data['avgwinddiection']=vancouver.groupby('date')['winddirection'].mean()

data['maxwinddirection']=vancouver.groupby('date')['winddirection'].max()

data['minwindirection']=vancouver.groupby('date')['winddirection'].min()
data.reset_index(level=0,inplace=True)
t=pd.DatetimeIndex(data['date'])

data['date']=t.date

data['year']=t.year

data['month']=t.month

data['day']=t.day
#Summarised data of Vancouver city dataset:

data.head()
data.shape
data.describe()
vancouver['description'].describe()
vancouver_new=vancouver.drop(['datetime','date','month','year','time'],axis=1)

vancouver_new.head()
vancouver_new=pd.get_dummies(vancouver_new)
from statsmodels.tsa.arima_model import ARIMA
#Assuming target variable as temperature:

m=ARIMA(vancouver_new['temperature'], order=(4,1,0)) # lags=4, displacement=1, moving average = 0
mf=m.fit(disp=0) # disp=o, to overcome lag error
print (mf.summary())
residual=DataFrame(mf.resid)
residual.plot()
#Predicting model for vancouver city weather data, assuming 'Temperature'  as target variable:

x=vancouver_new.drop(['temperature'],axis=1)

y=vancouver_new['temperature']

import sklearn.model_selection as model_selection

x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.30,random_state=200)
import sklearn.tree as tree

import sklearn.metrics as metrics

import sklearn.preprocessing as preprocessing

reg=tree.DecisionTreeRegressor(random_state=200)

model=model_selection.GridSearchCV(reg,param_grid={'max_depth':[3,4,5,6,7]})
model.fit(x_train,y_train)
model.best_estimator_
model.best_score_
#Found best estimator as 7:

reg=tree.DecisionTreeRegressor(max_depth=7)
reg.fit(x_train,y_train)
reg.score(x_test,y_test)
reg.feature_importances_
#Important features influencing the target variable 'temperature:

pd.Series(reg.feature_importances_,index=x.columns).sort_values(ascending=False)
x_test=preprocessing.normalize(x_test)

print (metrics.mean_squared_error(y_test,reg.predict(x_test)))
#Predicting model for vancouver city weather data, assuming 'Pressure'  as target variable:

x1=vancouver_new.drop(['pressure'],axis=1)

y1=vancouver_new['pressure']

x1_train,x1_test,y1_train,y1_test=model_selection.train_test_split(x1,y1,test_size=0.30,random_state=200)
reg=tree.DecisionTreeRegressor(random_state=200)

model1=model_selection.GridSearchCV(reg,param_grid={'max_depth':[3,4,5,6,7]})
model1.fit(x1_train,y1_train)

model1.best_estimator_
model1.best_score_
#Found best estimator as 5:

reg1=tree.DecisionTreeRegressor(max_depth=5)

reg1.fit(x1_train,y1_train)

reg1.score(x1_test,y1_test)
reg1.feature_importances_
#Important features influencing the target variable 'pressure':

pd.Series(reg1.feature_importances_,index=x1.columns).sort_values(ascending=False)
x1_test=preprocessing.normalize(x1_test)

print (metrics.mean_squared_error(y1_test,reg1.predict(x1_test)))
#Predicting model for vancouver city weather data, assuming 'windspeed'  as target variable:

x2=vancouver_new.drop(['windspeed'],axis=1)

y2=vancouver_new['windspeed']

x2_train,x2_test,y2_train,y2_test=model_selection.train_test_split(x2,y2,test_size=0.30,random_state=200)
reg=tree.DecisionTreeRegressor(random_state=200)

model2=model_selection.GridSearchCV(reg,param_grid={'max_depth':[3,4,5,6,7]})
model2.fit(x2_train,y2_train)

model2.best_estimator_
#Found best estimator as 7:

reg2=tree.DecisionTreeRegressor(max_depth=7)

reg2.fit(x2_train,y2_train)

reg2.score(x2_test,y2_test)
reg2.feature_importances_
#Important features influencing the target variable 'wind speed':

pd.Series(reg2.feature_importances_,index=x2.columns).sort_values(ascending=False)
reg2.predict(x2_test)
x2_test=preprocessing.normalize(x2_test)

print (metrics.mean_squared_error(y2_test,reg2.predict(x2_test)))
data=data.drop(['date','month','year'],axis=1)

data.head()
#Predicting model for vancouver city summarised weather data, assuming ' Average Temperature'  as target variable:

x3=data.drop(['avgtemp'],axis=1)

y3=data['avgtemp']

x3_train,x3_test,y3_train,y3_test=model_selection.train_test_split(x3,y3,test_size=0.30,random_state=200)
reg=tree.DecisionTreeRegressor(random_state=200)

model3=model_selection.GridSearchCV(reg,param_grid={'max_depth':[3,4,5,6,7]})
model3.fit(x3_train,y3_train)

model3.best_estimator_
#Found best estimator as 7:

reg3=tree.DecisionTreeRegressor(max_depth=7)

reg3.fit(x3_train,y3_train)

reg3.score(x3_test,y3_test)
reg3.feature_importances_
#Important features influencing the target variable 'average temperature':

pd.Series(reg3.feature_importances_,index=x3.columns).sort_values(ascending=False)
x3_test=preprocessing.normalize(x3_test)

print (metrics.mean_squared_error(y3_test,reg3.predict(x3_test)))
import sklearn.preprocessing as preprocessing

vancouver_scaled=preprocessing.scale(vancouver_new,axis=0)
vancouver[vancouver['year']==2016].tail()
v1=vancouver.drop(['datetime','date','month','year','time','description'],axis=1)

#Assuming period 'T' as 2016_12, so train dataset contains data upto 31stDecember 2016, remaining as test dataset:

train=v1[:37260]

test=v1[37260:]
from statsmodels.tsa.vector_ar.var_model import VAR
model4=VAR(endog=train)
model4_fit=model4.fit()
prediction=model4_fit.forecast(model4_fit.y,steps=len(test))
pred=pd.DataFrame(index=range(0,len(prediction)),columns=[v1.columns])

for i in range(0,6):

    for j in range(0,len(prediction)):

        pred.iloc[j][i]=prediction[j][i]
pred.head()
#MSE values:

import sklearn.metrics as metrics

for i in v1.columns:

    print ('MSE value for',i, 'is:', metrics.mean_squared_error(pred[i],test[i]))
#Fitting the model on complete dataset:

model5=VAR(endog=v1)

model5_fit=model5.fit()
predict=model5_fit.forecast(model5_fit.y,steps=1)
print (predict)