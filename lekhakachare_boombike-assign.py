# import all libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

#to calculaate vif
from statsmodels.stats.outliers_influence import variance_inflation_factor


import warnings
warnings.filterwarnings("ignore")
#read dataset
df=pd.read_csv("../input/day-dataset/DAY.csv")

# Column "instant" has unique value for each row, so we can drop it
df.drop("instant",axis=1,inplace=True)   
df.head()
# check shape of dataset
df.shape
#check null values and datatype of columns.
df.info()
# identify continuous and categorical columns
df.nunique().sort_values()
# analyzie continuous columns
df.describe()
# check values of season column
df["season"].value_counts()
#convert feature values into categorical string values of season column
#(1:spring, 2:summer, 3:fall, 4:winter)
df["season"].replace((1,2,3,4),("spring","summer","fall","winter"),inplace=True)
#check values of weathersit column
df["weathersit"].value_counts()
#convert feature values into categorical string values of weathersit column
#1: Clear, Few clouds, Partly cloudy, Partly cloudy 
#2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
#3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
df["weathersit"].replace((1,2,3),("clear","mist+claudy","light snow rain"),inplace=True)
#convert feature value of month into categorical string values
df["mnth"].replace((1,2,3,4,5,6,7,8,9,10,11,12),("Jan","Feb","March","April","May","Jun","July","Aug","Sep",
                                                "Oct","Nov","Dec"),inplace=True)
#convert feature value of weekday into categorical string values
df["weekday"].replace((0,1,2,3,4,5,6),("Sun","Mon","Tue","Wed","Thus","Fri","Sat"),inplace=True)
df.head()
sns.pairplot(df[["yr","temp","atemp","hum","windspeed","cnt"]])
plt.show()
plt.figure(figsize=(20,20))

plt.subplot(4,2,1)
sns.boxplot(data=df,x="yr",y="cnt")

plt.subplot(4,2,2)
sns.boxplot(data=df,x="mnth",y="cnt")

plt.subplot(4,2,3)
sns.boxplot(data=df,x="holiday",y="cnt")

plt.subplot(4,2,4)
sns.boxplot(data=df,x="weekday",y="cnt")

plt.subplot(4,2,5)
sns.boxplot(data=df,x="workingday",y="cnt")

plt.subplot(4,2,6)
sns.boxplot(data=df,x="season",y="cnt")

plt.subplot(4,2,7)
sns.boxplot(data=df,x="weathersit",y="cnt")

plt.show()
#find correlation between all columns 
df.corr()
df.drop(["casual","registered","atemp","dteday"],axis=1,inplace=True)
df.head()
# create dummy variables of mnth,season,weathersit,weekday columns
status=pd.get_dummies(df["mnth"],drop_first=True)
status1=pd.get_dummies(df["season"],drop_first=True)
status2=pd.get_dummies(df["weathersit"],drop_first=True)
status3=pd.get_dummies(df["weekday"],drop_first=True)
# merge dataset with dummy variables data set
df=pd.concat([df,status,status1,status2,status3],axis=1)
# drop columns
df.drop(["mnth","season","weathersit","weekday"],axis=1,inplace=True)
df.head()
# check shape of df
df.shape
# train_test_split
df_train,df_test=train_test_split(df,train_size=0.70,random_state=100)
# check shape of training set
df_train.shape
# check shape of test set
df_test.shape
df_train.head()
# rescaling(using Min-Max scaling)
scaler=MinMaxScaler()
num_var=["temp","hum","windspeed","cnt"]
df_train[num_var]=scaler.fit_transform(df_train[num_var])
df_train.head()
df_train[num_var].describe()
# find correlation between variables
plt.figure(figsize=(30,20))
df_train.corr()
sns.heatmap(df_train.corr(),annot=True,cmap="YlGnBu")
plt.show()
#make X_train, y_train
y_train=df_train.pop("cnt")
X_train=df_train
# make training model
lm=LinearRegression()
lm.fit(X_train,y_train)

#fit RFE
rfe=RFE(lm,15)
rfe=rfe.fit(X_train,y_train)
# ckeck columns of X_train
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# X_train columns with rfe supoort
col=X_train.columns[rfe.support_]
col
ncol=X_train.columns[~rfe.support_]
ncol
# creating X_train dataframe with rfe selected variables
X_train_rfe=X_train[col]
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)


# create first linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_1=lr.fit()

#check summary
lr_model_1.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build second model without "Jan"

X_train_rfe.drop("Jan",axis=1,inplace=True)
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create 2nd linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_2=lr.fit()

#check summary
lr_model_2.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build third model without "holiday"
X_train_rfe.drop("holiday",axis=1,inplace=True)
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create 3rd linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_3=lr.fit()

#check summary
lr_model_3.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build fourth model without "Spring"
X_train_rfe.drop("spring",axis=1,inplace=True)
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create 4th linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_4=lr.fit()

#check summary
lr_model_4.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build fifth model without "July"
X_train_rfe.drop("July",axis=1,inplace=True)
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create 5th linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_5=lr.fit()

#check summary
lr_model_5.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build fifth model without "light snow rain"
X_train_rfe.drop("light snow rain",axis=1,inplace=True)
# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create 6th linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_6=lr.fit()

#check summary
lr_model_6.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# build model without "mist+claudy" 
X_train_rfe.drop("mist+claudy",axis=1,inplace=True)

# add constant
X_train_rfe=sm.add_constant(X_train_rfe)

# create first linear regression model
lr=sm.OLS(y_train,X_train_rfe)

#fit model
lr_model_7=lr.fit()

#check summary
lr_model_7.summary()
# checking VIF
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# final models columns
X_train_rfe.columns
#create y_predicted values
y_train_pred=lr_model_7.predict(X_train_rfe)
# find out residual values
res=y_train-y_train_pred
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot(res)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.show()
# create list of numerical columns
num_var=["temp","hum","windspeed","cnt"]

# rescaling( converting values of continuous columns between 0 and 1)
df_test[num_var]=scaler.transform(df_test[num_var])
df_test.head()
df_test.describe()
# make x_test and y_test
y_test=df_test.pop("cnt")
X_test=df_test
# add constant
X_test_sm=sm.add_constant(X_test)
#remove columns which are remove in train data set
X_test_sm=X_test_sm.drop(['Aug', 'Dec', 'Feb', 'Jun', 'March', 'May', 'Nov', 'Oct', 'Mon', 'Sun',
       'Thus', 'Tue', 'Wed',"Jan","July","spring","holiday","mist+claudy","light snow rain"], axis=1)
#prediction

y_test_pred=lr_model_7.predict(X_test_sm)
# evaluate:
r2_test=r2_score(y_true=y_test,y_pred=y_test_pred)
r2_test
#Adj_R_squared
# formula:  Adj_R_squared=1-(1-R2)*(n-1)/(n-p-1)

#n =sample size=1 
#p = number of independent variables=9

Adj_R_squared=1-(1-r2_test)*(10-1)/(10-1-1)
Adj_R_squared
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
sns.regplot(y_test, y_test_pred,line_kws={"color": "red"})
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  
plt.show()
