# Supress Warnings
import warnings
warnings.filterwarnings('ignore')
#Import the relevant libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
pd.set_option('display.max_rows', None )
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importing bike.csv
bike = pd.read_csv('/kaggle/input/daycsv/day.csv')
# Display first few rows of application data
bike.head()
# Data Dimensions
bike.shape
# Display null value percentage of bike data
display(round(100*(bike.isnull().sum()/len(bike.index)), 2))
# Display datatypes
display(bike.info())  
# Dropping unnecessary and redundant columns
bike.drop(["instant","dteday", "casual", "registered"],axis =1, inplace = True) 
bike.head()
# Data Dimensions
bike.shape
# list of columns
bike.columns
# Checking datatypes 
bike.info()
# describing variables
bike.describe()
numerical = ['temp', 'atemp', 'hum', 'windspeed',]
plt.figure(figsize =(12,6))
for i in enumerate(numerical):
    plt.subplot(2,2, i[0]+1)
    sns.boxplot(x= i[1], data = bike)
df_num = bike[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]
sns.pairplot(df_num,corner=True)
plt.show()
sns.distplot(bike['temp'],hist = False, label= 'temp')
sns.distplot(bike['atemp'],hist = False, label= 'atemp')
plt.show()
plt.figure(figsize=(20,4))
plt.subplot(131)
sns.countplot(x= 'season', data = bike)
plt.subplot(132)
sns.countplot(x= 'mnth', data = bike)
plt.subplot(133)
sns.countplot(x= 'weekday', data = bike)
plt.show()
plt.figure(figsize=(16,4))
plt.subplot(141)
sns.countplot(x= 'yr', data = bike)
plt.subplot(142)
sns.countplot(x= 'holiday', data = bike)
plt.subplot(143)
sns.countplot(x= 'workingday', data = bike)
plt.subplot(144)
sns.countplot(x= 'weathersit', data = bike)
plt.show()
plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
sns.boxplot(x = 'yr', y = 'cnt', data = bike)
plt.subplot(2,2,2)
sns.boxplot(x = 'holiday', y = 'cnt', data = bike)
plt.subplot(2,2,3)
sns.boxplot(x = 'workingday', y = 'cnt', data = bike)
plt.subplot(2,2,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bike)
plt.show()
plt.figure(figsize=(16, 4))
plt.subplot(121)
sns.boxplot(x = 'season', y = 'cnt', data = bike)
plt.subplot(122)
sns.boxplot(x = 'weekday', y = 'cnt', data = bike)
plt.show()

plt.figure(figsize=(10,4))
sns.boxplot(x = 'mnth', y = 'cnt', data = bike)
plt.show()
plt.figure(figsize = (16,10))        # Size of the figure
sns.heatmap(bike.corr(),annot = True,cmap="YlGnBu")
plt.show()
# Check the housing dataframe now
bike.head()
bike["season"].value_counts()
bike["yr"].value_counts()
bike["mnth"].value_counts()
bike["holiday"].value_counts()
bike["weekday"].value_counts()
bike["workingday"].value_counts()
bike["weathersit"].value_counts()
bike_new = pd.get_dummies(data=bike, columns=['weathersit', 'weekday', 'mnth', 'season'], drop_first = True)
bike_new.head()
bike_new.info()
# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(bike_new, train_size = 0.70, random_state = 42)
df_train.shape
df_test.shape
bike_new.shape
scaler = MinMaxScaler()
df_train.head()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['temp', 'hum', 'windspeed','cnt','atemp']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
# Describing data
df_train.describe()
# Plotting heatmap of dataframe
plt.figure(figsize = (20, 20))
sns.heatmap(round(df_train.corr(),2), annot = True, cmap="YlGnBu")
plt.show()
# Dividing into X and Y sets for the model building
import copy
train_plot = copy.copy(df_train)
y_train = df_train.pop('cnt')
X_train = df_train
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
# List of variables after applying RFE
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# List of variables to be considered for model building
col = X_train.columns[rfe.support_]
col
# List of variables to be removed from data set
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
X_train_vif = X_train[col]
# Adding a constant variable 
X_train_rfe = sm.add_constant(X_train_rfe)
lm_1 = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
# Summary of our linear model
print(lm_1.summary())
# Defining function to check vif
def vif_show(X_vif):
    vif = pd.DataFrame()
    vif['Features'] = X_vif.columns
    vif['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
vif_show(X_train_vif)
X_train_rfe = X_train_rfe.drop(["atemp"], axis = 1)
X_train_vif = X_train_vif.drop(["atemp"], axis = 1)
col_to_drop = ['atemp']
lm_2 = sm.OLS(y_train,X_train_rfe).fit()  
print(lm_2.summary())
vif_show(X_train_vif)
X_train_rfe = X_train_rfe.drop(["hum"], axis = 1)
X_train_vif = X_train_vif.drop(["hum"], axis = 1)
col_to_drop.append('hum')
lm_3 = sm.OLS(y_train,X_train_rfe).fit()  
print(lm_3.summary())
vif_show(X_train_vif)
X_train_rfe = X_train_rfe.drop(["season_3"], axis = 1)
X_train_vif = X_train_vif.drop(["season_3"], axis = 1)
col_to_drop.append('season_3')
lm_4 = sm.OLS(y_train,X_train_rfe).fit()  
print(lm_4.summary())
vif_show(X_train_vif)
X_train_rfe = X_train_rfe.drop(["mnth_8"], axis = 1)
X_train_vif = X_train_vif.drop(["mnth_8"], axis = 1)
col_to_drop.append('mnth_8')
lm_5 = sm.OLS(y_train,X_train_rfe).fit()  
print(lm_5.summary())
vif_show(X_train_vif)
X_train_rfe = X_train_rfe.drop(["mnth_7"], axis = 1)
X_train_vif = X_train_vif.drop(["mnth_7"], axis = 1)
col_to_drop.append('mnth_7')
lm_6 = sm.OLS(y_train,X_train_rfe).fit()  
print(lm_6.summary())
vif_show(X_train_vif)
y_train_pred = lm_6.predict(X_train_rfe)
# Plotting Error terms of train data
fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)# X-label
plt.show()
res = y_train - y_train_pred
#res = res.abs()
plt.scatter(y_train_pred,res)
plt.xlabel('predicted value', fontsize = 18)
plt.ylabel('residuals', fontsize = 18)
plt.show()
df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('cnt')
X_test = df_test
# Dropping unwanted columns
X_test_new = X_test[col]
X_test_new.drop(col_to_drop, axis = 1 , inplace = True)
# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)
# Making predictions
y_pred = lm_6.predict(X_test_new)
# Calculating test data R^2
r2 = r2_score(y_test, y_pred)
print(r2)
# n is number of rows in X
n = X_test_new.shape[0]
# Number of features (predictors, p) is the shape along axis 1
p = X_test_new.shape[1]
# We find the Adjusted R-squared using the formula
adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
plt.show()