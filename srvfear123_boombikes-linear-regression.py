import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("/kaggle/input/boombikes/day.csv")
df.head()
#Total 730 rows with 16 columns

df.shape
#No missing values, dteday is not needed and we shall drop it later
df.info()
#We can see the data doesnt have any outliers
df.describe(percentiles=[0.75,0.9,0.99])
#We can se some high correlation between atemp and temp ofcourse, seasons and month because they are assigned with ranked values.
#Correlation between casual, registered and cnt are high because cnt is the sum of casual and registered

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True)
plt.show()
#Lets drop the date and the instant column
df = df.drop(['instant','dteday'],axis=1)
df.head()
#Lets check for some linearity.Like we see temp and atemp are strong linear relationship while atemp and temp are highly correlated
sns.pairplot(df[['temp','atemp','hum','windspeed','cnt']])

plt.scatter(x='temp',y='cnt',data=df)
plt.show()


plt.scatter(x='temp',y='casual',data=df)
plt.show()


plt.scatter(x='temp',y='registered',data=df)
plt.show()

plt.scatter(x='atemp',y='cnt',data=df)
plt.show()


plt.scatter(x='atemp',y='casual',data=df)
plt.show()


plt.scatter(x='atemp',y='registered',data=df)
plt.show()

plt.scatter(x='hum',y='cnt',data=df)
plt.show()


plt.scatter(x='hum',y='casual',data=df)
plt.show()


plt.scatter(x='hum',y='registered',data=df)
plt.show()

plt.scatter(x='windspeed',y='cnt',data=df)
plt.show()


plt.scatter(x='windspeed',y='casual',data=df)
plt.show()


plt.scatter(x='windspeed',y='registered',data=df)
plt.show()
#Seasons are showing some linearity, while the users fall during winter season, specially the casual users


sns.barplot(x=df['season'],y=df['cnt'])
plt.show()

sns.barplot(x=df['season'],y=df['casual'])
plt.show()

sns.barplot(x=df['season'],y=df['registered'])
plt.show()
#As expected, winter months show low users

sns.barplot(x=df['mnth'],y=df['cnt'])
plt.show()

sns.barplot(x=df['mnth'],y=df['casual'])
plt.show()

sns.barplot(x=df['mnth'],y=df['registered'])
plt.show()
#We can see the weekday showing some linearity with total users.
#Some important observations can be obtained here:
#like casual users decrease during weekday and increase when weekend approaches while the registered users drop significantly in weekends, this signifies registered users are mostly working professionals/employees

sns.barplot(x=df['weekday'],y=df['cnt'])
plt.show()

sns.barplot(x=df['weekday'],y=df['casual'])
plt.show()

sns.barplot(x=df['weekday'],y=df['registered'])
plt.show()
#As expected, casual users are high on non working days while registered users are high on working days
sns.barplot(x=df['workingday'],y=df['cnt'])
plt.show()

sns.barplot(x=df['workingday'],y=df['casual'])
plt.show()

sns.barplot(x=df['workingday'],y=df['registered'])
plt.show()
#2019 shows significant increase in users

sns.barplot(x=df['yr'],y=df['cnt'])
plt.show()

sns.barplot(x=df['yr'],y=df['casual'])
plt.show()

sns.barplot(x=df['yr'],y=df['registered'])
plt.show()
#We see weathersit type 1 has the users
sns.barplot(x=df['weathersit'],y=df['cnt'])
plt.show()

sns.barplot(x=df['weathersit'],y=df['casual'])
plt.show()

sns.barplot(x=df['weathersit'],y=df['registered'])
plt.show()
#Weathersit type 4 is not present in the data
df['weathersit'].unique()
df['weekday'] = df['weekday'].apply(lambda x : 0 if x==6 or x==0 else 1)
#Lets check the proportion of registered users in weekends

df_weekend = df[df['weekday'] ==0]
df_weekend
#We can see on weekends, 68% of total users are registered. We expect this to be more on weekdays
(df_weekend['registered'].sum()/df_weekend['cnt'].sum())*100
#Not much of difference between mean and median, so it is safe to assume the average percentage
print(df_weekend['casual'].median())
print(df_weekend['casual'].mean())

print(df_weekend['registered'].median())
print(df_weekend['registered'].mean())
#Similarly we check registered users for weekdays

df_weekday = df[df['weekday'] ==1]
df_weekday
#As expected, registered users are more on weekdays. Like we are sure, registered users are mostly working class

(df_weekday['registered'].sum()/df_weekday['cnt'].sum())*100
#We can safely assume average percentage since no outliers
print(df['casual'].median())
print(df['casual'].mean())

print(df['registered'].median())
print(df['registered'].mean())
#lets assign a new variable to our dataframe

df_new = df
#Assigning names to the seasons because numbers will create weights within
season_dict = {1:'Spring',2:'Summer',3:'Fall',4:'Winter'}
df_new['season'] = df_new['season'].map(season_dict)
df_new.head()

#Same goes for the month
month_dict = {1:'Jan',2:'Feb',3:'Mar',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
df_new['mnth'] = df_new['mnth'].map(month_dict)
df_new.head()
#Same for weathersit, type 4 is not present in dataset but let's just maintain the integrity
#CFPP-Clear, Few clouds, Partly cloudy, Partly cloudy
#MCBMM- Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
#LLTSL- Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
#HITS- Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog


weathersit_dict = {1:'CFPP',2:'MCBMM',3:'LLTSL',4:'HITS'}
df_new['weathersit'] = df_new['weathersit'].map(weathersit_dict)
df_new.head()
#Creating dummies and removing one column to reduce redundancy and same goes for all the categorical columns

season_dummies = pd.get_dummies(df_new['season'],drop_first=True,prefix='season')
df_new = pd.concat((df_new,season_dummies),axis=1)
df_new.head()
month_dummies = pd.get_dummies(df_new['mnth'],drop_first=True,prefix='month')
df_new = pd.concat((df_new,month_dummies),axis=1)
df_new.head()
weather_dummies = pd.get_dummies(df_new['weathersit'],drop_first=True,prefix='weather')
df_new = pd.concat((df_new,weather_dummies),axis=1)
df_new.head()
#lets drop the original columns since we have created the respective dummies

df_new = df_new.drop(['season','mnth','weathersit'],axis=1)
df_new.head()
X = df_new.drop(['cnt'],axis=1)
y = df['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_test = sm.add_constant(X_test)
X_train = sm.add_constant(X_train)
#We will be standardizing the columns with their respective Z scores

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#We fit and transform the train dataset on the required columns

X_train[['atemp','temp','hum','windspeed']] = scaler.fit_transform(X_train[['atemp','temp','hum','windspeed']])
X_train.head()
#We transform the test dataset based on training data

X_test[['atemp','temp','hum','windspeed']] = scaler.transform(X_test[['atemp','temp','hum','windspeed']])
X_test.head()
#Creating the regression object from scikit learn's regression package
lr = LinearRegression()
#Fitting the model with RFE package
lr.fit(X_train.iloc[:,1:],y_train)
rfe = RFE(lr,15)
rfe = rfe.fit(X_train.iloc[:,1:],y_train)
list(zip(X_train.iloc[:,1:].columns,rfe.support_,rfe.ranking_))
#We we list top 15 variables.
#We include const since its present in the dataset already

var = ['const','yr','holiday','weekday','workingday','temp','hum','windspeed','season_Spring','season_Summer','season_Winter','month_Jan','month_July','month_Sep','weather_LLTSL','weather_MCBMM']
#Fitting the model with statsmodels for better summary results

lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
var = ['const','yr','weekday','holiday','temp','hum','windspeed','season_Spring','season_Summer','season_Winter','month_Jan','month_July','month_Sep','weather_LLTSL','weather_MCBMM']
lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
var = ['const','yr','holiday','weekday','temp','windspeed','season_Spring','season_Summer','season_Winter','month_Jan','month_July','month_Sep','weather_LLTSL','weather_MCBMM']
lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
var = ['const','yr','holiday','weekday','temp','windspeed','season_Spring','season_Summer','season_Winter','month_July','month_Sep','weather_LLTSL','weather_MCBMM']
lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
var = ['const','yr','holiday','temp','windspeed','season_Spring','season_Summer','season_Winter','month_July','month_Sep','weather_LLTSL','weather_MCBMM']
lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
plt.figure(figsize=(10,7))
sns.heatmap(X_train[var].iloc[:,1:].corr(),annot=True)
plt.show()
var = ['const','yr','holiday','temp','windspeed','season_Summer','season_Spring','season_Winter','month_Sep','weather_LLTSL','weather_MCBMM']
lr = sm.OLS(y_train,X_train[var]).fit()
lr.summary()
vif = pd.DataFrame()
vif['Features'] = X_train[var].iloc[:,1:].columns
vif['VIF'] = [variance_inflation_factor(X_train[var].iloc[:,1:].values,i) for i in range(X_train[var].iloc[:,1:].shape[1])]
vif = vif.sort_values(by='VIF',ascending=False)
vif
#We predict for X_train

prediction_train = lr.predict(X_train[var])
#We get an R2 of 83.2% which is good
r2_score(y_train,prediction_train)
#Lets apply the normal proportion for registered users since it showed more linearity than casual with 86% on weekdays and 68% on weekends
X_train['registered_train'] = np.where(X_train['weekday'] == 1, prediction_train * 0.86, prediction_train * 0.68)
#Approximately 82% of predicted registered users are explainable which is again good
r2_score(X_train['registered_train'],X_train['registered'])
#The errors are normally distributed which makes our assumption right
sns.distplot(y_train-prediction_train)
plt.scatter(y=y_train-prediction_train,x=y_train)
#We get R2 of 81.9% for test data which is good. This shows our model is stable. 
#PS- This R2 will be different from R2_score below because here the test data is being treated as a training data and the data is fit with the statsmodels

lr_1 = sm.OLS(y_test,X_test[var]).fit()
lr_1.summary()
#Lets predict for the test data based from training data

prediction_test = lr.predict(X_test[var])
prediction_test
# Lets check the R2_score from the result based from training data.
r2_score(y_test,prediction_test)
#Lets apply the normal proportion for registered users since it showed more linearity than casual with 86% on weekdays and 68% on weekends
X_test['registered_test'] = np.where(X_test['weekday'] == 1, prediction_test * 0.86, prediction_test * 0.68)
#Approximately 78% of predicted registered users are explainable which is again good
r2_score(X_test['registered'],X_test['registered_test'])