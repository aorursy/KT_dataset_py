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
# Importing all required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import datetime



# Supress Warnings

import warnings

warnings.filterwarnings('ignore')
#Import data into python

boombikes = pd.read_csv('../input/bike-sharing-assignment/day.csv')
# Check head of the Data

boombikes.head()
#Check info 

boombikes.info()
# Check shape

boombikes.shape
# Check describe

boombikes.describe()
#Check columnwise null values

boombikes.isnull().mean()
# check row-wise null values

boombikes.isnull().mean(axis=1)
boombikes.isnull().values.any()
# Create a dummy dataframefor duplicate check

bb_duplicates = boombikes



# Checking for duplicates and dropping the entire duplicate row if any

bb_duplicates.drop_duplicates(subset=None, inplace=True)

print(bb_duplicates.shape)

print(boombikes.shape)
boombikes.drop(['instant','dteday', 'casual','registered'], axis=1, inplace=True)
boombikes.head()
# Check the correlation between the variables in Dataframe

plt.figure(figsize = (16, 10))

sns.heatmap(boombikes.corr(),annot = True, cmap="YlGnBu")

plt.show()
#Drop atemp variable as we are using temp variable in analysis

boombikes.drop(['atemp'], axis = 1, inplace = True)

boombikes.head()
plt.figure(figsize = (16, 10))

sns.heatmap(boombikes.corr(),annot = True, cmap="YlGnBu")

plt.show()
#Check data types of the variable

boombikes.dtypes
# Convert categoric variables to 'category' data type



boombikes['season']=boombikes['season'].astype('category')

boombikes['weathersit']=boombikes['weathersit'].astype('category')

boombikes['mnth']=boombikes['mnth'].astype('category')

boombikes['weekday']=boombikes['weekday'].astype('category')

boombikes.info()
#adding categorical values into categorical variables

boombikes['season'] = boombikes['season'].map({1:'spring', 2:'summer',3:'fall',4:'winter'})

boombikes['mnth'] = boombikes['mnth'].map({1:'JAN', 2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'})

boombikes['weekday'] = boombikes['weekday'].map({0:'SUN',1:'MON', 2:'TUE',3:'WED',4:'THU',5:'FRI',6:'SAT'})

boombikes['weathersit'] = boombikes['weathersit'].map({1:'Clear', 2:'Mist & Cloudy',3:'Light snow & Rain',4:'Heavy snow & rain'})
boombikes.head()
##### 5.1 Numeric Variables:

sns.boxplot('temp', data = boombikes)

plt.show()
sns.boxplot('hum', data = boombikes)

plt.show()
sns.boxplot('windspeed', data = boombikes)

plt.show()
sns.barplot(x ='yr', y = 'cnt' ,data = boombikes )
sns.barplot(x ='holiday', y = 'cnt' ,data = boombikes )
sns.barplot(x ='workingday', y = 'cnt' ,data = boombikes )
sns.barplot(x ='season', y = 'cnt' ,data = boombikes )
sns.barplot(x ='mnth', y = 'cnt' ,data = boombikes )
sns.barplot(x ='weekday', y = 'cnt' ,data = boombikes )
sns.barplot(x ='weathersit', y = 'cnt' ,data = boombikes )
#Get Dummy variables for 'season'

season = pd.get_dummies(boombikes['season'], drop_first = True)

boombikes = pd.concat([boombikes, season], axis = 1)

#Get Dummy variables for 'mnth'

mnth = pd.get_dummies(boombikes['mnth'], drop_first = True)

boombikes = pd.concat([boombikes, mnth], axis = 1)

#Get Dummy variables for 'weekday'

weekday = pd.get_dummies(boombikes['weekday'], drop_first = True)

boombikes = pd.concat([boombikes, weekday], axis = 1)

#Get Dummy variables for 'weathersit'

weathersit = pd.get_dummies(boombikes['weathersit'], drop_first = True)

boombikes = pd.concat([boombikes, weathersit], axis = 1)

boombikes.head()

#Drop the variables for which dummies are created.

boombikes.drop(['season','mnth','weekday','weathersit'], axis=1, inplace = True)

boombikes.head()
boombikes.shape
boombikes.dtypes
#Import train_test_split from sklearn model_selection

from sklearn.model_selection import train_test_split
train, test = train_test_split(boombikes, train_size = 0.7, random_state = 100)
print(train.shape)

print(test.shape)

print(boombikes.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['temp', 'hum', 'windspeed','cnt']



train[num_vars] = scaler.fit_transform(train[num_vars])
sns.heatmap(train[num_vars].corr(),annot = True, cmap="YlGnBu")

plt.show()
train.head()
train.describe()
y_train = train.pop('cnt')

X_train = train
X_train.head()
y_train.head()
#Importing the statsmodels api 

import statsmodels.api as sm

# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
#Running RFE

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)

rfe = rfe.fit(X_train, y_train)



list(zip(X_train.columns,rfe.support_,rfe.ranking_))

#Featured columns after rfe

col = X_train.columns[rfe.support_]



print(col)

print(X_train.columns[~rfe.support_])
##lm_1

# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]

X_train_rfe = sm.add_constant(X_train_rfe)

lm_1 = sm.OLS(y_train,X_train_rfe).fit()

print(lm_1.summary())
#Import variance_inflation_factor from statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



X_train_new = X_train_rfe.drop(['const'], axis=1)

# Calculate the VIFs 

vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
##Lm_2

#New model without "fall"



X_train_new = X_train_new.drop(["fall"], axis = 1)

X_train_lm = sm.add_constant(X_train_new)

lm_2 = sm.OLS(y_train,X_train_lm).fit()

print(lm_2.summary())
# VIFs without "fall"

# X_train_new = X_train_new.drop(['const'], axis =1)

vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#New model without "fall"& "hum"

X_train_new = X_train_new.drop(["hum"], axis = 1)

X_train_lm = sm.add_constant(X_train_new)

lm_3 = sm.OLS(y_train,X_train_lm).fit()

print(lm_3.summary())
# VIFs without "fall"& "hum"



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#New model without "fall"& "hum" & "OCT"

X_train_new = X_train_new.drop(["OCT"], axis = 1)

X_train_lm = sm.add_constant(X_train_new)

lm_4 = sm.OLS(y_train,X_train_lm).fit()

print(lm_4.summary())
# VIFs without "fall"& "hum" & "OCT"



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#New model without  "fall"& "hum" & "OCT" & "holiday"

X_train_new = X_train_new.drop(["holiday"], axis = 1)

X_train_lm = sm.add_constant(X_train_new)

lm_5 = sm.OLS(y_train,X_train_lm).fit()

print(lm_5.summary())
# VIFs without "fall"& "holiday" & "OCT" & "workingday"



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#New model without "fall","hum","OCT","holiday","AUG"

X_train_new6 = X_train_rfe.drop(["fall","hum","OCT","holiday","AUG"], axis = 1)

X_train_lm6 = sm.add_constant(X_train_new6)

lm_6 = sm.OLS(y_train,X_train_lm6).fit()

print(lm_6.summary())

print(lm_6.params)
# VIFs without 

X_train_new6  = X_train_new6.drop(["const"], axis =1)

vif = pd.DataFrame()

X = X_train_new6

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_pred = lm_6.predict(X_train_lm6)
res = y_train-y_train_pred



# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
# Perform scaling on test data for numeric variables

num_vars = ['temp', 'hum', 'windspeed','cnt']

test[num_vars] = scaler.fit_transform(test[num_vars])
test.describe()
test.head()
#Dividing into X_test and y_test

y_test = test.pop('cnt')

X_test = test
X_test.head()
X_test.info()
#Selecting the variables that were part of final model.

final_columns = X_train_new6.columns



X_test=X_test[final_columns]



# Adding constant variable to test dataframe

X_test_lm6 = sm.add_constant(X_test)



X_test_lm6.info()
X_test_lm6.head()
# Making predictions using the final model (lm_7)



y_pred = lm_6.predict(X_test_lm6)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)             

plt.xlabel('y_test', fontsize = 18)                         

plt.ylabel('y_pred', fontsize = 16) 
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
X_test.shape
# n is number of rows in X

r2= r2_score(y_test, y_pred)

n = X_test.shape[0]





# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2