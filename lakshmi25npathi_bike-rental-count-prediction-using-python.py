import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



#import the csv file

bike_df=pd.read_csv("../input/day.csv")
#Shape of the dataset

bike_df.shape
#Data types

bike_df.dtypes
#Read the data

bike_df.head(5)
#Rename the columns

bike_df.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',

                       'hum':'humidity','cnt':'total_count'},inplace=True)
#Read the data

bike_df.head(5)
#Type casting the datetime and numerical attributes to category



bike_df['datetime']=pd.to_datetime(bike_df.datetime)



bike_df['season']=bike_df.season.astype('category')

bike_df['year']=bike_df.year.astype('category')

bike_df['month']=bike_df.month.astype('category')

bike_df['holiday']=bike_df.holiday.astype('category')

bike_df['weekday']=bike_df.weekday.astype('category')

bike_df['workingday']=bike_df.workingday.astype('category')

bike_df['weather_condition']=bike_df.weather_condition.astype('category')
#Summary of the dataset

bike_df.describe()
#Missing values in dataset

bike_df.isnull().sum()
fig,ax=plt.subplots(figsize=(15,8))

sns.set_style('white')

#Bar plot for seasonwise monthly distribution of counts

sns.barplot(x='month',y='total_count',data=bike_df[['month','total_count','season']],hue='season',ax=ax)

ax.set_title('Seasonwise monthly distribution of counts')

plt.show()

#Bar plot for weekday wise monthly distribution of counts

fig,ax1=plt.subplots(figsize=(15,8))

sns.barplot(x='month',y='total_count',data=bike_df[['month','total_count','weekday']],hue='weekday',ax=ax1)

ax1.set_title('Weekday wise monthly distribution of counts')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Violin plot for yearly distribution of counts

sns.violinplot(x='year',y='total_count',data=bike_df[['year','total_count']])

ax.set_title('Yearly distribution of counts')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Barplot for Holiday distribution of counts

sns.barplot(data=bike_df,x='holiday',y='total_count',hue='season')

ax.set_title('Holiday wise distribution of counts')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Bar plot for workingday distribution of counts

sns.barplot(data=bike_df,x='workingday',y='total_count',hue='season')

ax.set_title('Workingday wise distribution of counts')

plt.show()
fig,ax1=plt.subplots(figsize=(15,8))

#Bar plot for weather_condition distribution of counts

sns.barplot(x='weather_condition',y='total_count',data=bike_df[['month','total_count','weather_condition']],ax=ax1)

ax1.set_title('Weather_condition wise monthly distribution of counts')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Boxplot for total_count outliers

sns.boxplot(data=bike_df[['total_count']])

ax.set_title('total_count outliers')

plt.show()
fig,ax=plt.subplots(figsize=(15,8))

#Box plot for Temp_windspeed_humidity_outliers

sns.boxplot(data=bike_df[['temp','windspeed','humidity']])

ax.set_title('Temp_windspeed_humidity_outiers')

plt.show()
from fancyimpute import KNN



#create dataframe for outliers

wind_hum=pd.DataFrame(bike_df,columns=['windspeed','humidity'])

 #Cnames for outliers                     

cnames=['windspeed','humidity']       

                      

for i in cnames:

    q75,q25=np.percentile(wind_hum.loc[:,i],[75,25]) # Divide data into 75%quantile and 25%quantile.

    iqr=q75-q25 #Inter quantile range

    min=q25-(iqr*1.5) #inner fence

    max=q75+(iqr*1.5) #outer fence

    wind_hum.loc[wind_hum.loc[:,i]<min,:i]=np.nan  #Replace with NA

    wind_hum.loc[wind_hum.loc[:,i]>max,:i]=np.nan  #Replace with NA

#Imputating the outliers by mean Imputation

wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].mean())

wind_hum['humidity']=wind_hum['humidity'].fillna(wind_hum['humidity'].mean())
#Replacing the imputated windspeed

bike_df['windspeed']=bike_df['windspeed'].replace(wind_hum['windspeed'])

#Replacing the imputated humidity

bike_df['humidity']=bike_df['humidity'].replace(wind_hum['humidity'])

bike_df.head(5)
import scipy

from scipy import stats

#Normal plot

fig=plt.figure(figsize=(15,8))

stats.probplot(bike_df.total_count.tolist(),dist='norm',plot=plt)

plt.show()
#Create the correlation matrix

correMtr=bike_df[["temp","atemp","humidity","windspeed","casual","registered","total_count"]].corr()

mask=np.array(correMtr)

mask[np.tril_indices_from(mask)]=False

#Heat map for correlation matrix of attributes

fig,ax=plt.subplots(figsize=(15,8))

sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)

ax.set_title('Correlation matrix of attributes')

plt.show()
#load the required libraries

from sklearn import preprocessing,metrics,linear_model

from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split
#Split the dataset into the train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(bike_df.iloc[:,0:-3],bike_df.iloc[:,-1],test_size=0.3, random_state=42)



#Reset train index values

X_train.reset_index(inplace=True)

y_train=y_train.reset_index()



# Reset train index values

X_test.reset_index(inplace=True)

y_test=y_test.reset_index()



print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

print(y_train.head())

print(y_test.head())
#Create a new dataset for train attributes

train_attributes=X_train[['season','month','year','weekday','holiday','workingday','weather_condition','humidity','temp','windspeed']]

#Create a new dataset for test attributes

test_attributes=X_test[['season','month','year','weekday','holiday','workingday','humidity','temp','windspeed','weather_condition']]

#categorical attributes

cat_attributes=['season','holiday','workingday','weather_condition','year']

#numerical attributes

num_attributes=['temp','windspeed','humidity','month','weekday']
#To get dummy variables to encode the categorical features to numeric

train_encoded_attributes=pd.get_dummies(train_attributes,columns=cat_attributes)

print('Shape of transfomed dataframe::',train_encoded_attributes.shape)

train_encoded_attributes.head(5)
#Training dataset for modelling

X_train=train_encoded_attributes

y_train=y_train.total_count.values
#training model

lr_model=linear_model.LinearRegression()

lr_model
#fit the trained model

lr_model.fit(X_train,y_train)
#Accuracy of the model

lr=lr_model.score(X_train,y_train)

print('Accuracy of the model :',lr)

print('Model coefficients :',lr_model.coef_)

print('Model intercept value :',lr_model.intercept_)
#Cross validation prediction

predict=cross_val_predict(lr_model,X_train,y_train,cv=3)

predict
#Cross validation plot

fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y_train,y_train-predict)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

plt.show()
#R-squared scores

r2_scores = cross_val_score(lr_model, X_train, y_train, cv=3)

print('R-squared scores :',np.average(r2_scores))
#To get dummy variables to encode the categorical features to numeric

test_encoded_attributes=pd.get_dummies(test_attributes,columns=cat_attributes)

print('Shape of transformed dataframe :',test_encoded_attributes.shape)

test_encoded_attributes.head(5)
#Test dataset for prediction

X_test=test_encoded_attributes

y_test=y_test.total_count.values
#predict the model

lr_pred=lr_model.predict(X_test)

lr_pred
import math

#Root mean square error 

rmse=math.sqrt(metrics.mean_squared_error(y_test,lr_pred))

#Mean absolute error

mae=metrics.mean_absolute_error(y_test,lr_pred)

print('Root mean square error :',rmse)

print('Mean absolute error :',mae)
#Residual plot

fig, ax = plt.subplots(figsize=(15,8))

ax.scatter(y_test, y_test-lr_pred)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Observed')

ax.set_ylabel('Residuals')

ax.title.set_text("Residual Plot")

plt.show()
#training the model

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(min_samples_split=2,max_leaf_nodes=10)
#Fit the trained model

dtr.fit(X_train,y_train)
#Accuracy score of the model

dtr_score=dtr.score(X_train,y_train)

print('Accuracy of model :',dtr_score)
#Plot the learned model

from sklearn import tree

import pydot

import graphviz



# export the learned model to tree

dot_data = tree.export_graphviz(dtr, out_file=None) 

graph = graphviz.Source(dot_data) 

graph
predict=cross_val_predict(dtr,X_train,y_train,cv=3)

predict
# Cross validation prediction plot

fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y_train,y_train-predict)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

plt.show()
#R-squared scores

r2_scores = cross_val_score(dtr, X_train, y_train, cv=3)

print('R-squared scores :',np.average(r2_scores))
#predict the model

dtr_pred=dtr.predict(X_test)

dtr_pred
#Root mean square error

rmse=math.sqrt(metrics.mean_squared_error(y_test,dtr_pred))

#Mean absolute error

mae=metrics.mean_absolute_error(y_test,dtr_pred)

print('Root mean square error :',rmse)

print('Mean absolute error :',mae)
#Residual scatter plot

residuals = y_test-dtr_pred

fig, ax = plt.subplots(figsize=(15,8))

ax.scatter(y_test, residuals)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

ax.set_title('Residual plot')

plt.show()
#Training the model

from sklearn.ensemble import RandomForestRegressor

X_train=train_encoded_attributes

rf=RandomForestRegressor(n_estimators=200)
#Fit the trained model

rf.fit(X_train,y_train)
#accuracy of the model

rf_score =rf.score(X_train,y_train)

print('Accuracy of the model :',rf_score)
#Cross validation prediction

predict=cross_val_predict(rf,X_train,y_train,cv=3)

predict
#Cross validation prediction plot

fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y_train,y_train-predict)

ax.axhline(lw=2,color='black')

ax.set_title('Cross validation prediction plot')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

plt.show()
#R-squared scores

r2_scores = cross_val_score(rf, X_train, y_train, cv=3)

print('R-squared scores :',np.average(r2_scores))

#predict the model

X_test=test_encoded_attributes

rf_pred=rf.predict(X_test)

rf_pred
#Root mean square error

rmse = math.sqrt(metrics.mean_squared_error(y_test,rf_pred))

print('Root mean square error :',rmse)

#Mean absolute error

mae=metrics.mean_absolute_error(y_test,rf_pred)

print('Mean absolute error :',mae)
#Residual scatter plot

fig, ax = plt.subplots(figsize=(15,8))

residuals=y_test-rf_pred

ax.scatter(y_test, residuals)

ax.axhline(lw=2,color='black')

ax.set_xlabel('Observed')

ax.set_ylabel('Residuals')

ax.set_title('Residual plot')

plt.show()
Bike_df1=pd.DataFrame(y_test,columns=['y_test'])

Bike_df2=pd.DataFrame(rf_pred,columns=['rf_pred'])

Bike_predictions=pd.merge(Bike_df1,Bike_df2,left_index=True,right_index=True)

Bike_predictions.to_csv('Bike_Renting_Python.csv')

Bike_predictions