import warnings
warnings.filterwarnings('ignore')

from sklearn import preprocessing
from IPython.core.interactiveshell import InteractiveShell
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
# from collections import defaultdicta
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

%matplotlib inline
# READ DATA

bikedata = pd.read_csv("../input/boombikes/day.csv",parse_dates=['dteday']) 
print(bikedata.head())
#shape check 
print(bikedata.shape)
#  descriptive information check

print(bikedata.info())
#descriptive  statistical information check

print(bikedata.describe())
# percentage of missing values in each column

round(100*(bikedata.isnull().sum()/len(bikedata.index)), 2).sort_values(ascending=False)

bike_duplicate = bikedata

# Checking for duplicates and dropping the entire duplicate row if any
bike_duplicate.drop_duplicates(subset=None, inplace=True)
bike_duplicate.shape
bikedata.columns
bikedata_new=bikedata[['season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
       'cnt']]
bikedata_new.info()
#converting the datatype to category
bikedata_new['season']=bikedata_new['season'].astype('category')
bikedata_new['weathersit']=bikedata_new['weathersit'].astype('category')
bikedata_new['mnth']=bikedata_new['mnth'].astype('category')
bikedata_new['weekday']=bikedata_new['weekday'].astype('category')
bikedata_new.info()
#creating the dummy variables
#using drop_first to drop the first variable for each set of dummies created

bikedata_new = pd.get_dummies(bikedata_new, drop_first=True)

bikedata_new.info()
bikedata_new.head()
from sklearn.model_selection import train_test_split

np.random.seed(0)
df_train, df_test = train_test_split(bikedata_new, train_size = 0.70, test_size = 0.30, random_state = 333)
#checking out training set info
df_train.info()
#checking out training set size
df_train.shape
#checking out testing set info
df_test.info()
#checking out testing set size
df_test.shape

bikedata_num=df_train[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']] #taking only numerical variable

sns.pairplot(bikedata_num, diag_kind='kde')
plt.show()
#taking categorical variables before creating dummy variables

plt.figure(figsize=(25, 10))
plt.subplot(2,3,1)
sns.boxplot(x = 'season', y = 'cnt', data = bikedata)
plt.subplot(2,3,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = bikedata)
plt.subplot(2,3,3)
sns.boxplot(x = 'weathersit', y = 'cnt', data = bikedata)
plt.subplot(2,3,4)
sns.boxplot(x = 'holiday', y = 'cnt', data = bikedata)
plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = bikedata)
plt.subplot(2,3,6)
sns.boxplot(x = 'workingday', y = 'cnt', data = bikedata)
plt.show()
plt.figure(figsize = (25,20))
ax=sns.heatmap(bikedata_new.corr(), annot = True, cmap="YlGnBu")
bottom,top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
    
df_train.head()
df_train.columns
# Apply scaler()

numerical_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_train[numerical_vars] = scaler.fit_transform(df_train[numerical_vars])
# Checking values after scaling
df_train.head()
df_train.describe()
y_train = df_train.pop('cnt')
X_train = df_train
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             # running RFE
rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
col

X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]
#VIF CHECK 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
import statsmodels.api as sm

# Add a constant
X_train_lm1 = sm.add_constant(X_train_rfe)

# Create a first fitted model
lr1 = sm.OLS(y_train, X_train_lm1).fit()
# parameter check

lr1.params
#model Summary
print(lr1.summary())
X_train_new = X_train_rfe.drop(["atemp"], axis = 1)
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm2 = sm.add_constant(X_train_new)

# Create a first fitted model
lr2 = sm.OLS(y_train, X_train_lm2).fit()
# Check the parameters obtained

lr2.params
print(lr2.summary())
X_train_new = X_train_new.drop(["hum"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm3 = sm.add_constant(X_train_new)

# Create a first fitted model
lr3 = sm.OLS(y_train, X_train_lm3).fit()
lr3.params
print(lr3.summary())
X_train_new = X_train_new.drop(["season_3"], axis = 1)
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm4 = sm.add_constant(X_train_new)

# Create a first fitted model
lr4 = sm.OLS(y_train, X_train_lm4).fit()
lr4.params
print(lr4.summary())
X_train_new = X_train_new.drop(["mnth_10"], axis = 1)
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm5 = sm.add_constant(X_train_new)

# Create a first fitted model
lr5 = sm.OLS(y_train, X_train_lm5).fit()
lr5.params
print(lr5.summary())
X_train_new = X_train_new.drop(["mnth_3"], axis = 1)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Add a constant
X_train_lm6 = sm.add_constant(X_train_new)

# Create a first fitted model
lr6 = sm.OLS(y_train, X_train_lm6).fit()
lr6.params
print(lr6.summary())
lr6.params
y_train_predict = lr6.predict(X_train_lm6)
residual = y_train-y_train_predict


fig = plt.figure()
sns.distplot((residual), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)  
plt.xlabel('Errors', fontsize = 18)    
bikedata_new=bikedata_new[[ 'temp', 'atemp', 'hum', 'windspeed','cnt']]

sns.pairplot(bikedata_num, diag_kind='kde')
plt.show()
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] = X_train_new.columns
vif['VIF'] = [variance_inflation_factor(X_train_new.values, i) for i in range(X_train_new.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
#applying scaling

numerical_vars = ['temp', 'atemp', 'hum', 'windspeed','cnt']

df_test[numerical_vars] = scaler.transform(df_test[numerical_vars])
df_test.head()
df_test.describe()
#Dividing into X_test and y_test

y_test = df_test.pop('cnt')
X_test = df_test

X_test.info()
#Selecting the variables that are part of final model.
col1=X_train_new.columns

X_test=X_test[col1]

# Adding constant variable to test dataframe
X_test_lm6 = sm.add_constant(X_test)

X_test_lm6.info()
# Making predictions using the final model (lr6)

y_predict = lr6.predict(X_test_lm6)
# Plotting y_test and y_pred to understand the spread
# import matplotlib.pyplot as plt
# import numpy as np


fig = plt.figure()
plt.scatter(y_test, y_predict, alpha=.5)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16) 
#r2 = 1-(RSS/TSS)

from sklearn.metrics import r2_score
r2_score(y_test, y_predict)
r2=0.8203092200749708
# n is number of rows in X

n = X_test.shape[0]


# Number of features (predictors, p) is the shape along axis 1
p = X_test.shape[1]

# We find the Adjusted R-squared using the formula

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2
