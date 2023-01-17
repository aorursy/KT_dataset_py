# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
bike_sharing=pd.read_csv("/kaggle/input/boombikes/day.csv")
bike_sharing.head()
bike_sharing.shape
bike_sharing.info()
bike_sharing.describe()
plt.figure(figsize=(16,10))
sns.pairplot(bike_sharing)
plt.show()
bike_sharing['season']=bike_sharing['season'].map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})
bike_sharing.head()
dum_season = pd.get_dummies(bike_sharing['season'],drop_first = True)
dum_season.head()
bike_sharing['weathersit']=bike_sharing['weathersit'].map({1:'clear', 2:'mist', 3:'lightrain', 4:'heavyrain'})
bike_sharing.head()
dum_weathersit = pd.get_dummies(bike_sharing['weathersit'],drop_first = True)
dum_weathersit.head()
bike_sharing['mnth']=bike_sharing['mnth'].map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
bike_sharing.head()
dum_mnth = pd.get_dummies(bike_sharing['mnth'],drop_first = True)
dum_mnth.head()
bike_sharing['weekday']=bike_sharing['weekday'].map({0:'Sun',1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu',5:'Fri',6:'Sat'})
bike_sharing.head()
dum_weekday = pd.get_dummies(bike_sharing['weekday'],drop_first = True)
dum_weekday.head()
# concating the dummy varibles into the dataset.
bike_sharing = pd.concat([bike_sharing, dum_season,dum_weathersit,dum_mnth,dum_weekday], axis = 1)
bike_sharing.head()
# here we also drop registered and casual varibles as it is higly correlated with target varible count

bike_sharing.drop(['season','dteday','mnth','weekday','casual','registered','instant','weathersit','atemp'], axis = 1, inplace = True)
bike_sharing.head()
bike_sharing_train, bike_sharing_test = train_test_split(bike_sharing, train_size = 0.7, test_size = 0.3, random_state = 100)
bike_sharing_test.shape
bike_sharing_train.shape
scaler = MinMaxScaler()
# scaling the train dataset
num_vars = ['temp','hum','windspeed','cnt']

bike_sharing_train[num_vars] = scaler.fit_transform(bike_sharing_train[num_vars])
bike_sharing_train.head()
bike_sharing_train.describe()
plt.figure(figsize = (16, 10))
sns.heatmap(bike_sharing_train.corr(), annot = True, cmap="YlGnBu")
plt.show()
y_train = bike_sharing_train.pop('cnt') # target variable
X_train = bike_sharing_train

lr=LinearRegression()
lr.fit(X_train,y_train)
rfe=RFE(lr,15)
rfe=rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col=X_train.columns[rfe.support_]
col
X_train.columns[~rfe.support_]
# Assigning the feature variables having True value to X
X_train_rfe=X_train[col]
# Adding constants to X_train
X_train_lr=sm.add_constant(X_train_rfe)
# Create a first fitted model
lr=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables 
X_train_new=X_train_rfe.drop(['hum'],axis=1)
X_train_lr=sm.add_constant(X_train_new)
lr_1=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_1.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Dropping highly correlated variables as temp has high coefficient so we drop next variable with high VIF.

X_train_new1=X_train_new.drop(['workingday'],axis=1)
X_train_lr=sm.add_constant(X_train_new1)
lr_2=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_2.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new2=X_train_new1.drop(['Sat'],axis=1)
X_train_lr=sm.add_constant(X_train_new2)
lr_3=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_3.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new3=X_train_new2.drop(['Jan'],axis=1)
X_train_lr=sm.add_constant(X_train_new3)
lr_4=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_4.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new4=X_train_new3.drop(['summer'],axis=1)
X_train_lr=sm.add_constant(X_train_new4)
lr_5=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_5.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new4
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new5=X_train_new4.drop(['winter'],axis=1)
X_train_lr=sm.add_constant(X_train_new5)
lr_6=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_6.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new5
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
X_train_new6=X_train_new5.drop(['Sep'],axis=1)
X_train_lr=sm.add_constant(X_train_new6)
lr_7=sm.OLS(y_train,X_train_lr).fit()
# Print a summary of the linear regression model obtained
print(lr_7.summary())
# Check for the VIF values of the feature variables. 
vif = pd.DataFrame()
X=X_train_new6
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_cnt = lr_7.predict(X_train_lr)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)  
num_vars = ['temp','hum','windspeed','cnt']

bike_sharing_test[num_vars] = scaler.transform(bike_sharing_test[num_vars])
bike_sharing_test.describe()
y_test = bike_sharing_test.pop('cnt')
X_test = bike_sharing_test
# Adding constant variable to test dataframe
X_test_lr = sm.add_constant(X_test)
X_test_lr.head()
X_test_lr=X_test_lr.loc[:,['const','yr','holiday','temp','windspeed','spring','lightrain','mist','Jul']]
X_test_lr
# Making predictions using the seventh model
y_pred_cnt = lr_7.predict(X_test_lr)
fig = plt.figure()
plt.scatter(y_test, y_pred_cnt)
fig.suptitle('y_test vs y_pred', fontsize = 20)              
plt.xlabel('y_test', fontsize = 18)                          
plt.ylabel('y_pred', fontsize = 16) 
r2=r2_score(y_test, y_pred_cnt)
r2
