import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import statsmodels.api as sm
%matplotlib inline

from statsmodels.stats.outliers_influence import variance_inflation_factor 



train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
print(train.shape)
print(test.shape)
train.info()
#check for NULL values in the data.
train.isna().sum()
#lets extract the different date related features from the datetime object
train['hour']=[t.hour for t in pd.DatetimeIndex(train.datetime)]
train['day']=[t.dayofweek for t in pd.DatetimeIndex(train.datetime)]
train['month']=[t.month for t in pd.DatetimeIndex(train.datetime)]
train['year']=[t.year for t in pd.DatetimeIndex(train.datetime)]
train['quarter']=[t.quarter for t in pd.DatetimeIndex(train.datetime)]
#lets extract the different date related features from the datetime object
test['hour']=[t.hour for t in pd.DatetimeIndex(test.datetime)]
test['day']=[t.dayofweek for t in pd.DatetimeIndex(test.datetime)]
test['month']=[t.month for t in pd.DatetimeIndex(test.datetime)]
test['year']=[t.year for t in pd.DatetimeIndex(test.datetime)]
test['quarter']=[t.quarter for t in pd.DatetimeIndex(test.datetime)]
#drop the field datetime
train.drop(['datetime'],inplace=True,axis=1)
test.drop(['datetime'],inplace=True,axis=1)
#lets use the forward selection of variable approach.
#create models separately for registered and casual users as the parameters impact these two measures separately.
#lets start with a model that can predict the registered users mode accurately.
# set some hypothesis based on our initial understanding.we will validate them during the exploratory data analysis
#H1 : Registered users probably use the bikes more during the office hours compared to non office hours.
#H2 : Registered users count on the weekends will be lower compared to the weekdays and holidays
#H3 : Registered users count will increase as the time increases.
#H4 : people tend to use less bikes during high humidity
#H5 : demand will be more during the good climate
#H6 : demand will be low during high temperature
#H7 : low demand during the wind speed is high.
#lets confirm whether the hypothesis we have defined are correct by doing EDA.
#every hypotheis is like a story.we are going to add story to our model in terms of the parameters.
train.head()
#average demand for the registered and casual users
f,axis=plt.subplots(1,2,figsize=(15,6))
b1=sns.barplot(data=train,x='hour',y='registered',ax=axis[0])
b2=sns.barplot(data=train,x='hour',y='casual',ax=axis[1])
#EDA support the H1.lets use this variable in the model.
train_x1=train['hour']
train_R_y1=train['registered']
train_C_y1=train['casual']
#model to predict the registed users count
model_R_1=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_1.summary())
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#predict the Registered and casual users
predict_R_1=model_R_1.predict(train_x1)
predict_C_1=model_C_1.predict(train_x1)
predict_count_1=predict_R_1+predict_C_1
#visualize predict_count_1 and actual count
plt.Figure(figsize=(5,5))
plt.scatter(predict_count_1,train['count'])
plt.plot([0,1000],[0,1000],color='red')
plt.xlim=(-100,1000)
plt.ylim=(-100,1000)
plt.xlabel('prediction')
plt.ylabel('train_y')
plt.title('Linear Regression Model')
#define RMSLE
def rmsle(prediction,actual):
    log1=np.array([np.log(v+1) for v in prediction])
    log2=np.array([np.log(v+1) for v in actual])
    calc=(log1-log2)**2
    return np.sqrt(np.mean(calc))
rmsle(predict_count_1,train['count'])
# above is the approach I am takin for this project
# now lets consider the #registered users only and add more features and improve the R squared.
#H2 : Registered users count on the weekends will be lower compared to the weekdays and holidays

sns.barplot(data=train,x='day',y='registered')


sns.barplot(data=train,x='workingday',y='registered')
# EDA proves our hypothesis. On working days the demand is more for registered users.
train_x1=train[['hour','workingday']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
#H3 : Registered users count will increase as the time increases.

#lets create a field for identifying the quarter
train['Quarter_number']=np.where((train['year']==2011) & (train['month']<=3),1,
                  np.where((train['year']==2011) & (train['month']>3) & (train['month']<=6),2,
                         np.where((train['year']==2011) & (train['month']>6) & (train['month']<=9),3,
                                np.where((train['year']==2011) & (train['month']>9) & (train['month']<=12),4,
                                       np.where((train['year']==2012) & (train['month']<3) ,5,
                                              np.where((train['year']==2012) & (train['month']>3) & (train['month']<=6),6,
                                                     np.where((train['year']==2012) & (train['month']>6) & (train['month']<=9),7,
                                                            np.where((train['year']==2012) & (train['month']>9) & (train['month']<=12),8,0
                                                                    ))))))))



#lets create a field for identifying the quarter in the test data set
test['Quarter_number']=np.where((test['year']==2011) & (test['month']<=3),1,
                  np.where((test['year']==2011) & (test['month']>3) & (test['month']<=6),2,
                         np.where((test['year']==2011) & (test['month']>6) & (test['month']<=9),3,
                                np.where((test['year']==2011) & (test['month']>9) & (test['month']<=12),4,
                                       np.where((test['year']==2012) & (test['month']<3) ,5,
                                              np.where((test['year']==2012) & (test['month']>3) & (test['month']<=6),6,
                                                     np.where((test['year']==2012) & (test['month']>6) & (test['month']<=9),7,
                                                            np.where((test['year']==2012) & (test['month']>9) & (test['month']<=12),8,0
                                                                    ))))))))


train_x1=train[['hour','workingday','Quarter_number']]
train_R_y1=train['registered']
train.Quarter_number.unique()
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
#The R squared value has increased to .627.
#H6 : demand will be low during high temperature
#lets do an EDA to check how temperature affect the demand

plt.scatter(x='humidity',y='registered',data=train)
#lets create buckets for humidity
train['humiditybins']=np.floor(train['humidity'])//5
train['tempbins']=np.floor(train['temp'])//5
train['atempbins']=np.floor(train['atemp'])//5
train['windspeedbins']=np.floor(train['windspeed'])//5

#lets create buckets for humidity
test['humiditybins']=np.floor(test['humidity'])//5
test['tempbins']=np.floor(test['temp'])//5
test['atempbins']=np.floor(test['atemp'])//5
test['windspeedbins']=np.floor(test['windspeed'])//5

sns.barplot(data=train,x='humiditybins',y='registered')
sns.barplot(data=train,x='windspeedbins',y='registered')
sns.barplot(data=train,x='tempbins',y='registered')
sns.barplot(data=train,x='atempbins',y='registered')
train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','atempbins','windspeedbins']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
#lets check the VIF value to see if any multicollinearity
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#looks like the vif factor for atempbins is high.lets remove and check the model
train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
sns.barplot(data=train,x='weather',y='registered')
#lets create dummy variables for weather
weather_1=pd.get_dummies(train['weather'],prefix='weather')
train=pd.concat([train,weather_1],axis=1)
train.head()
#lets create dummy variables for weather
weather_1=pd.get_dummies(test['weather'],prefix='weather')
test=pd.concat([test,weather_1],axis=1)
test.head()
train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins','weather_1','weather_2','weather_3','weather_4']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
#H5 : demand will be more during the good climate

vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#remove weather_4
train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins','weather_1','weather_2','weather_3']]
train_R_y1=train['registered']

#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())

vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#remove weather_2
train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins','weather_1','weather_3']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#lets create dummy variables for season
season_1=pd.get_dummies(train['season'],prefix='season')
train=pd.concat([train,season_1],axis=1)
train.head()
#lets create dummy variables for season
season_1=pd.get_dummies(test['season'],prefix='season')
test=pd.concat([test,season_1],axis=1)
test.head()

train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins','weather_1','weather_3','season_1','season_2','season_3','season_4']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif

train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','windspeedbins','weather_1','weather_3','season_1','season_2','season_4']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())

vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif

train_x1=train[['hour','workingday','Quarter_number','humiditybins','tempbins','weather_1','weather_3','season_1','season_2','season_4']]
train_R_y1=train['registered']
#model to predict the registed users count
model_R_2=sm.OLS(train_R_y1,train_x1).fit()
print(model_R_2.summary())

test_x1=test[['hour','workingday','Quarter_number','humiditybins','tempbins','weather_1','weather_3','season_1','season_2','season_4']]


#predict the Registered and casual users
predict_R_2=model_R_2.predict(train_x1)

#visualize predict_count_1 and actual count
plt.Figure(figsize=(5,5))
plt.scatter(predict_R_2,train['registered'])
plt.plot([0,1000],[0,1000],color='red')
plt.xlim=(-100,1000)
plt.ylim=(-100,1000)
plt.xlabel('prediction')
plt.ylabel('train_y')
plt.title('Linear Regression Model')
#predict the Registered and casual users
predict_R_2_test=model_R_2.predict(test_x1)


predict_R_2=np.where(predict_R_2<0,10,predict_R_2)
rmsle(predict_R_2,train['registered'])
predict_R_2_test=np.where(predict_R_2_test<0,10,predict_R_2_test)

rmsle(predict_R_2,train['registered'])
##############Now lets try to improve the model for casual customers###############
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#H6 : casual users count has no impact on the office hours

sns.barplot(data=train,x='hour',y='casual')
#the hypothesis is true
train_x1=train['hour']
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#now lets see the casual users count on the days

sns.barplot(data=train,x='day',y='casual')
#casual users count on the weekends is high.
sns.barplot(data=train,x='workingday',y='casual')
train_x1=train[['hour','workingday']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#R squared has improved after adding the working day variable
#H7 : lets see how the casual users count changes with time
sns.barplot(data=train,x='Quarter_number',y='casual')
train_x1=train[['hour','workingday','Quarter_number']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#lets see how the temperature,humidity and windspeed is affecting the demand
sns.barplot(data=train,x='humiditybins',y='casual')
sns.barplot(data=train,x='atempbins',y='casual')
sns.barplot(data=train,x='tempbins',y='casual')
sns.barplot(data=train,x='windspeedbins',y='casual')
train_x1=train[['hour','workingday','Quarter_number','windspeedbins','atempbins','tempbins','humiditybins']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#R squared is now increased to .639
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#lets remove atempbins
train_x1=train[['hour','workingday','Quarter_number','windspeedbins','tempbins','humiditybins']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
#lets check how weather affect the casual customer
sns.barplot(data=train,x='weather',y='casual')
#lets remove atempbins
train_x1=train[['hour','workingday','Quarter_number','windspeedbins','tempbins','humiditybins','weather_1','weather_2','weather_3','weather_4']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#lets remove atempbins
train_x1=train[['hour','workingday','Quarter_number','windspeedbins','tempbins','humiditybins','weather_1','weather_2','weather_3']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif
#lets remove windspeed as it is not relevant
train_x1=train[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
sns.barplot(data=train,x='season',y='casual')


train_x1=train[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3','season_1','season_2','season_3','season_4']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif

train_x1=train[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3','season_1','season_2','season_3']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif

train_x1=train[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3','season_2','season_3']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
vif=pd.DataFrame()
vif['viffactor']=[variance_inflation_factor(train_x1.values,i) for i in range(train_x1.shape[1])]
vif['features']=train_x1.columns
vif

train_x1=train[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3','season_3']]
train_C_y1=train['casual']
#model to predict the casual users count
model_C_1=sm.OLS(train_C_y1,train_x1).fit()
print(model_C_1.summary())
test_x1=test[['hour','workingday','Quarter_number','tempbins','humiditybins','weather_1','weather_2','weather_3','season_3']]
#predict the Registered and casual users
predict_C_1=model_C_1.predict(train_x1)

#visualize predict_count_1 and actual count
plt.Figure(figsize=(5,5))
plt.scatter(predict_C_1,train['casual'])
plt.plot([0,500],[0,500],color='red')
plt.xlim=(-100,500)
plt.ylim=(-100,500)
plt.xlabel('prediction')
plt.ylabel('train_y')
plt.title('Linear Regression Model')
predict_C_2=np.where(predict_C_1<0,10,predict_C_1) #predicted casual users count
train_x1.info()
test_x1.info()
predict_C_1_test=model_C_1.predict(test_x1) #predicting the the casual users count on test data
predict_C_2_test=np.where(predict_C_1_test<0,10,predict_C_1_test) #convert the -ve numbers to +ve

predict_C_2_test
rmsle(predict_C_2,train['casual']) #RMSLE for the predicted casual users count on the train.
rmsle(predict_R_2,train['registered']) #RMSLE for the predicted registered users count on the train
final_predict_count=predict_C_2+predict_R_2 #final count for the train data set
final_predict_count_test=predict_C_2_test+predict_R_2_test #final count for the test data set
final_predict_count_test
rmsle(final_predict_count,train['count'])
final_predict_count_test.shape

test=pd.read_csv("../input/test.csv")
test.info()
d={'datetime':test['datetime'],'count':final_predict_count_test}
ans=pd.DataFrame(d)
ans.to_csv('answer.csv',index=False) # saving to a csv file for predictions on kaggle.
