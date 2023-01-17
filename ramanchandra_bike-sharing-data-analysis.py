import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sn

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))



Bike_df=pd.read_csv("../input/day.csv")

Bike_df.shape

Bike_df.head(10)
Bike_df.dtypes
#clean up attribute names

Bike_df.rename(columns={'instant':'rec_id',

                        'dteday':'datetime',

                        'holiday':'is_holiday',

                        'workingday':'is_workingday',

                       'weathersit':'weather_condition',

                       'hum':'humidity',

                       'mnth':'month',

                       'cnt':'total_count',

                       'yr':'year'},inplace=True)
Bike_df.head(10)
#date time conversion

Bike_df['datetime']=pd.to_datetime(Bike_df.datetime)



# Categorical variables

Bike_df['season']=Bike_df.season.astype('category')

Bike_df['is_holiday']=Bike_df.is_holiday.astype('category')

Bike_df['is_workingday']=Bike_df.is_workingday.astype('category')

Bike_df['weekday']=Bike_df.weekday.astype('category')

Bike_df['weather_condition']=Bike_df.weather_condition.astype('category')

Bike_df['month']=Bike_df.month.astype('category')

Bike_df['year']=Bike_df.year.astype('category')

#Descriptive statistics for each column

np.round(Bike_df.describe(),2)
# Visualizing monthly raidershp counts across the seasons

fig,ax=plt.subplots(figsize=(20,8))

sn.pointplot(data=Bike_df[['month','total_count','season']],x='month',

             y='total_count',

             hue='season',ax=ax)

ax.set(title='Season wise montly distribution of raidership counts ')
fig,ax=plt.subplots(figsize=(20,8))

sn.pointplot(data=Bike_df[['month','total_count','weekday']],x='month',y='total_count',

            hue='weekday',ax=ax)

ax.set(title='Weekday wise monthly distribution of raidership counts')
fig,ax=plt.subplots(figsize=(20,8))

sn.barplot(data=Bike_df[['month','total_count']],x='month',y='total_count',ax=ax)

ax.set(title='Month-wise raidership distribution')
# Violin plot is used for Year-wise distribution

fig,ax=plt.subplots(figsize=(20,8))

sn.violinplot(data=Bike_df[['year','total_count']],x='year',y='total_count',ax=ax)

ax.set(title='Year-wise distribution of raidership counts')

fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(20,8))

sn.boxplot(data=Bike_df[['total_count','casual','registered']],ax=ax1)

sn.boxplot(data=Bike_df[['temp','windspeed','humidity']],ax=ax2)
# Correlation matrix to find correlations between data attributes

CorrMat=Bike_df[['temp','atemp','humidity','windspeed',

                 'casual','registered','total_count']].corr()

mask=np.array(CorrMat)

mask[np.tril_indices_from(mask)]=False

#Heat map to plot the Correlation matrix

fig,ax=plt.subplots(figsize=(20,8))

sn.heatmap(CorrMat,mask=mask,

          vmax=0.8,square=True,annot=True,ax=ax)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(Bike_df.iloc[:,0:-3],Bike_df.iloc[:,-1],

                                  test_size=0.3, random_state=42)

#Reset train index values

X_train.reset_index(inplace=True)

y_train=y_train.reset_index()

# Reset train index values

X_test.reset_index(inplace=True)

y_test=y_test.reset_index()

train_attributes=X_train[['season','month','year','weekday','is_holiday','is_workingday','weather_condition',

                         'humidity','temp','windspeed']]

test_attributes=X_test[['season','month','year','weekday','is_holiday','is_workingday',

                       'humidity','temp','windspeed','weather_condition']]

cat_attributes=['season','is_holiday','is_workingday','weather_condition','year']

num_attributes=['temp','windspeed','humidity','month','weekday']
#Transform categorical Variables

Bike_train=pd.get_dummies(train_attributes,columns=cat_attributes)

print('Shape of transfomed dataframe::',Bike_train.shape)

Bike_train.head(5)

X=Bike_train

y=y_train.total_count.values
from sklearn import linear_model

lin_reg=linear_model.LinearRegression()

lr_model=lin_reg.fit(X,y)

print('R-squared score for training dataset::',np.round(lr_model.score(X,y),3))

print('Model coefficients::',np.round(lr_model.coef_,3))

print('Model intercept value::',np.round(lr_model.intercept_,3))



# 3 fold cross validation (cv=3)

from sklearn.model_selection import cross_val_predict,cross_val_score

predicted=cross_val_predict(lr_model,X,y,cv=3)
fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y,y-predicted)

ax.axhline(lw=2,color='black')

ax.set_title('Residual plot for Cross validation')

ax.set_xlabel('Observed')

ax.set_ylabel('Residual')

plt.show()
#R-squared

r2_score=cross_val_score(lr_model,X,y,cv=3)

R2_score=np.average(r2_score)

print('R-squared Score for cross validation dataset::',np.round(R2_score,2))
# Tranform categorical data to numerical data for tesing dataset

Bike_test=pd.get_dummies(test_attributes,columns=cat_attributes)

print('Shape of transformed dataframe::',Bike_test.shape)

Bike_test.head(5)

from sklearn.metrics import r2_score

# Test dataset for model testing

X_t=Bike_test

y_t=y_test.total_count.values

#predict the model performance

y_pred=lr_model.predict(X_t)

residuals=y_t-y_pred

#predicted score

print('R-squared score for testing dataset::',np.round(r2_score(y_t,y_pred),3))

fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y_t, residuals)

ax.axhline(lw=2,color='black')

ax.set_title('Residual plot for testing dataset')

ax.set_xlabel('Observed')

ax.set_ylabel('Resduals')

plt.show()
X_d=Bike_train

y_d=y_train.total_count.values

X_d.shape,y_d.shape
#Decision tree regressor

from sklearn.tree import DecisionTreeRegressor

dtr=DecisionTreeRegressor(max_depth=5,min_samples_split=8,max_leaf_nodes=40,min_samples_leaf=3)

dtr.fit(X_d,y_d)

print('R-squared score::',np.round(dtr.score(X_d,y_d),2))
from sklearn import tree

import pydot

import graphviz

dtr_data=tree.export_graphviz(dtr,out_file=None)

dtr_graph=graphviz.Source(dtr_data)

dtr_graph
from sklearn.metrics import r2_score

# Test dataset for model testing

X_dt=Bike_test

y_dt=y_test.total_count.values

#predict the model performance

y_dpred=dtr.predict(X_dt)

residuals=y_dt-y_dpred

#predicted score

print('R-squared score for testing dataset::',np.round(r2_score(y_dt,y_dpred),3))
residuals=y_dt-y_dpred

fig,ax=plt.subplots(figsize=(15,8))

ax.scatter(y_dt, residuals)

ax.axhline(lw=2,color='black')

ax.set_title('Residual plot for testing dataset')

ax.set_xlabel('Observed')

ax.set_ylabel('Resduals')

plt.show()
# Train dataset

X_rf_tr=Bike_train

y_rf_tr=y_train.total_count.values

X_rf_tr.shape,y_rf_tr.shape
#Test dataset

X_rf_ts=Bike_test

y_rf_ts=y_test.total_count.values

X_rf_ts.shape,y_rf_ts.shape
# import the model

from sklearn.ensemble import RandomForestRegressor

# Assign no. of decision trees = 1000

rf=RandomForestRegressor(n_estimators=1000, random_state=42)

# Train the model on the training data

rf.fit(X_rf_tr,y_rf_tr)
# R-squared score for trained data

print('R-squared score for trained dataset::',np.round(rf.score(X_rf_tr,y_rf_tr),3))
#Use the forest's predict method

y_pred_rf=rf.predict(X_rf_ts)



# R-squared score for predictions

print('R-squared score for predictions::',np.round(r2_score(y_rf_ts,y_pred_rf),3))
#Dataframe for models

Model = ['Linear Regression','Decision tree','Random Forest']

df1=pd.DataFrame(Model,columns=['Model'])

R2_score = [0.846,0.84,0.897]

df2=pd.DataFrame(R2_score,columns=['R2_score'])

Model_df=pd.merge(df1,df2,left_index=True,right_index=True)

print('Model evaluation on test data:\n',Model_df)