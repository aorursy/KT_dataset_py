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



os.chdir('../input')

train=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')
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

#train.drop(['datetime'],inplace=True,axis=1)

#test.drop(['datetime'],inplace=True,axis=1)
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
#H1 : Registered users probably use the bikes more during the office hours compared to non office hours.

f,axis=plt.subplots(1,2,figsize=(15,6))

b1=sns.barplot(data=train,x='hour',y='registered',ax=axis[0])

b2=sns.barplot(data=train,x='hour',y='casual',ax=axis[1])
#EDA support the H1.there are more number of registered users during the office hours.Casual users demand is high during the non

#office hours.
#define RMSLE.This is the evaluation parameter to check the error.

def rmsle(prediction,actual):

    log1=np.array([np.log(v+1) for v in prediction])

    log2=np.array([np.log(v+1) for v in actual])

    calc=(log1-log2)**2

    return np.sqrt(np.mean(calc))
#H2 : Registered users count on the weekends will be lower compared to the weekdays and holidays





f,axis=plt.subplots(1,2,figsize=(10,4))

b1=sns.barplot(data=train,x='day',y='registered',ax=axis[0])

b2=sns.barplot(data=train,x='day',y='casual',ax=axis[1])

#Hypothesis is true.There are less number of people in the weekends as registered users.
f,axis=plt.subplots(1,2,figsize=(15,6))

b1=sns.barplot(data=train,x='workingday',y='registered',ax=axis[0])

b2=sns.barplot(data=train,x='workingday',y='casual',ax=axis[1])
# EDA proves our hypothesis. On working days the demand is more for registered users.
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

f,axis=plt.subplots(1,2,figsize=(15,6))

b1=sns.barplot(data=train,x='Quarter_number',y='registered',ax=axis[0])

b2=sns.barplot(data=train,x='Quarter_number',y='casual',ax=axis[1])
#registered users count clearly showing an incresing pattern as the quarters progress. Casual users also there is an overall

#increase in deamnd as the quarters progress.
#H4 : people tend to use less bikes during high humidity

#H6 : demand will be low during high temperature and low temperature

#H7 : low demand during the wind speed is high.

#lets do an EDA to confirm our analysis

fig,(ax1,ax2,ax3,ax4)=plt.subplots(ncols=4)

fig.set_size_inches(12,5)

sns.regplot(x='temp',y='count',data=train,ax=ax1)

sns.regplot(x='atemp',y='count',data=train,ax=ax2)

sns.regplot(x='humidity',y='count',data=train,ax=ax3)

sns.regplot(x='windspeed',y='count',data=train,ax=ax4)

sns.distplot

#regplot will plot data and a regression model fit.
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

#lets drop the original columns

train.drop('humidity',axis=1,inplace=True)

test.drop('humidity',axis=1,inplace=True)

train.drop('temp',axis=1,inplace=True)

test.drop('temp',axis=1,inplace=True)

train.drop('atemp',axis=1,inplace=True)

test.drop('atemp',axis=1,inplace=True)

train.drop('windspeed',axis=1,inplace=True)

test.drop('windspeed',axis=1,inplace=True)
#H5 : Now check the hypothesis about the weather.Demand is high during good weather



f,axis=plt.subplots(1,2,figsize=(10,4))

b1=sns.barplot(data=train,x='weather',y='registered',ax=axis[0])

b2=sns.barplot(data=train,x='weather',y='casual',ax=axis[1])
#registered users demand is high during weather 4 but the casual users demand is low for the same weather.
#now lets check the field season

fig,(ax1,ax2,ax3)=plt.subplots(ncols=3)

fig.set_size_inches(12,5)

sns.barplot(data=train,x='season',y='casual',ax=ax1)

sns.barplot(data=train,x='season',y='registered',ax=ax2)

sns.barplot(data=train,x='season',y='count',ax=ax3)
train.info()
test.info()
train.head()
#Now lets do the feature engineering with our understanding from the EDA
#lets create dummy variables for season,weather,hour,day,month,year,Quarter_number
def dummies(train,test,columns):

    for column in columns:

        train[column]=train[column].apply(lambda x:str(x))

        test[column]=test[column].apply(lambda x:str(x))

        good_cols=[column+'_'+i for i in train[column].unique() if i in test[column].unique()]

        train=pd.concat((train,pd.get_dummies(train[column],prefix=column)[good_cols]),axis=1)

        test=pd.concat((test,pd.get_dummies(test[column],prefix=column)[good_cols]),axis=1)

        del train[column]

        del test[column]

    return train,test
train_data,test_data=dummies(train,test,columns=['season','weather','hour','day','month','year','Quarter_number','humiditybins','tempbins','atempbins','windspeedbins'])
train_data.head()
test_data.head()
#drop the column quarter

train_data.drop(['datetime','quarter'],axis=1,inplace=True)

test_data.drop(['datetime','quarter'],axis=1,inplace=True)
#drop the columns casual and registered as well.

train_data.drop(['casual','registered'],axis=1,inplace=True)
#Now we are done with the feature engineering. Lets move onto the model building.
#lets divide the train dataset into train and test dataset.



#set Rseed

RSEED=70
import numpy as np



#labels are the values we need to predict

labels=np.array(train_data['count'])



#remove the labels from the features

features=train_data.drop('count',axis=1)



#saving feature names for later use

feature_list=list(features.columns)



#convert to numpy array

features=np.array(features)
#Training and Testing Sets

# Using Skicit-learn to split data into training and testing sets



from sklearn.model_selection import train_test_split

#split the data to training and testing set

train_features,test_features,train_labels,test_labels=train_test_split(features,labels,test_size=0.25,random_state=42)
print('Training features shape',train_features.shape)

print('Training labels shape',train_labels.shape)

print('Testing features shape',test_features.shape)

print('Testing labels shape',test_labels.shape)
#Train the model

#import the model we are using

from sklearn.ensemble import RandomForestRegressor

#Instantiate model with 1000 decision trees

rf=RandomForestRegressor(n_estimators=1000,random_state=42)
#train the model on the training data

rf.fit(train_features,train_labels)
#Make predictions on the test set

#use the predict method on the test data

predictions=rf.predict(test_features)
#calculate the RMSLE

rmsle(predictions,test_labels)
#lets check the feature importances.

importances=list(rf.feature_importances_)
#list of tuples with variable and importance

feature_importances=[(feature,round(importance,2)) for feature,

                    importance in zip(feature_list,importances)]
#sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
#lets make prediction on the test dataset.
features_test=np.array(features)
predictions_test=rf.predict(test_data)
test=pd.read_csv('test.csv')
d={'datetime':test['datetime'],'count':predictions_test}
ans=pd.DataFrame(d)
#ans.to_csv('answer.csv',index=False) # saving to a csv file for predictions on kaggle.