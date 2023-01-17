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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None, "display.max_columns", None)
bikes = pd.read_csv("/kaggle/input/bike-sharing-analysis-post-covid19/day.csv")

bikes.head()
bikes.info()
print("shape of dataset is" ,bikes.shape)
bikes.describe()
bikes.dtypes
round(100*(bikes.isnull().sum()/len(bikes)),2)
bikes.columns
drop_list = ['instant','dteday']

bikes_new = bikes.drop(drop_list, axis=1)
bikes_new.head()
bikes_new.info()
bikes_new['season']=bikes_new['season'].map({1:"spring", 2:"summer", 3:"fall", 4:"winter"})

bikes_new['season'].value_counts()
bikes_new['weathersit']=bikes_new['weathersit'].map({1:"clear", 2:"mist+cloudy", 3:"light_rain/snow", 4:"heavy_rain/snow"})

bikes_new['weathersit'].value_counts()
bikes_new['weekday']=bikes_new['weekday'].map({0:"Sunday", 1:"Monday", 2:"Tuesday", 3:"Wednesday",4:"Thursday", 5:"Friday", 6:"Saturday"})

bikes_new['weekday'].value_counts()
bikes_new['mnth']=bikes_new['mnth'].map({1:"Jan", 2:"Feb", 3:"Mar",4:"Apr", 5:"May", 6:"Jun",7:"Jul", 8:"Aug", 9:"Sep",10:"Oct", 11:"Nov", 12:"Dec"})

bikes_new['mnth'].value_counts()
bikes_new.info()
bikes_num=bikes_new[['temp','atemp', 'hum', 'windspeed','casual','registered','cnt']]

sns.pairplot(bikes_num, diag_kind='kde')

plt.show()
plt.figure(figsize=(25, 15))

plt.subplot(2,3,1)

sns.boxplot(x = 'season', y = 'cnt', data = bikes_new)

plt.subplot(2,3,2)

sns.boxplot(x = 'mnth', y = 'cnt', data = bikes_new)

plt.subplot(2,3,3)

sns.boxplot(x = 'weathersit', y = 'cnt', data = bikes_new)

plt.subplot(2,3,4)

sns.boxplot(x = 'holiday', y = 'cnt', data = bikes_new)

plt.subplot(2,3,5)

sns.boxplot(x = 'weekday', y = 'cnt', data = bikes_new)

plt.subplot(2,3,6)

sns.boxplot(x = 'workingday', y = 'cnt', data = bikes_new)

plt.show()
plt.figure(figsize = (25,15))

ax = sns.heatmap(bikes_new.corr(),square = True,annot=True, cmap="Reds")

bottom, top = ax.get_ylim()

ax.set_ylim(bottom + 0.5, top - 0.5);
bikes_new = bikes_new.drop(["casual","registered"], axis=1)

bikes_new.head()
bikes_new = pd.get_dummies(bikes_new, drop_first=True)

bikes_new.head()
from sklearn.model_selection import train_test_split



df_train, df_test = train_test_split(bikes_new, train_size = 0.7, test_size = 0.3, random_state = 14)
print("Shape of bike dataset:",bikes_new.shape)

print("Shape of training dataset :",df_train.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
num_vars = ['temp','atemp','hum','windspeed','cnt']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

df_train.head()
df_train.describe()
y_train = df_train.pop('cnt')

X_train = df_train
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 15)             

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
X_train_rfe = X_train[col]

X_train_rfe.head()
from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

print(lm.summary())
#Removing the const from the dataframe and creating a new dataframe as "X_train_new"



X_train_new = X_train_rfe.drop(["const",], axis = 1)
X_train_new = X_train_new.drop(["atemp",], axis = 1)
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm2 = sm.add_constant(X_train_new)



# Creating and running the linear model

lr2 = sm.OLS(y_train, X_train_lm2).fit()

print(lr2.summary())
X_train_new = X_train_new.drop(["windspeed"], axis = 1)
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm3 = sm.add_constant(X_train_new)



# Creating and running the linear model

lr3 = sm.OLS(y_train, X_train_lm3).fit()

print(lr3.summary())
X_train_new = X_train_new.drop(["mnth_Feb"], axis = 1)
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm4 = sm.add_constant(X_train_new)



# Creating and running the linear model

lr4 = sm.OLS(y_train, X_train_lm4).fit()

print(lr4.summary())
X_train_new = X_train_new.drop(["mnth_Jan"], axis = 1)
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm5 = sm.add_constant(X_train_new)



# Creating and running the linear model

lr5 = sm.OLS(y_train, X_train_lm5).fit()

print(lr5.summary())
X_train_new = X_train_new.drop(["hum"], axis = 1)
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Add a constant

X_train_lm6 = sm.add_constant(X_train_new)



# Creating and running the linear model

lr6 = sm.OLS(y_train, X_train_lm6).fit()

print(lr6.summary())
lr6.params
y_train_pred = lr6.predict(X_train_lm6)
res = y_train-y_train_pred

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((res), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18) 
vif = pd.DataFrame()

vif['Features'] = X_train_new.columns

vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
num_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']



df_test[num_vars] = scaler.transform(df_test[num_vars])
df_test.describe()
y_test = df_test.pop('cnt')

X_test = df_test

X_test.info()
#Selecting the variables that were part of final model.

col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe

X_test_lm6 = sm.add_constant(X_test)

X_test_lm6.info()
# Making predictions using the final model (lr6)

y_pred = lr6.predict(X_test_lm6)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred, alpha=.5)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16) 

plt.show()
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
r2=0.8199435928434153
# n is number of rows in X

n = X_test.shape[0]



# Number of features (predictors, p) is the shape along axis 1

p = X_test.shape[1]



# We find the Adjusted R-squared using the formula

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2