# Supress warnings
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
data = pd.read_csv("../input/boombikes/day.csv")

# Checking the head of the dataset
data.head()
data.shape
data.info()
data.describe()
#dropping all variables which not related to the count of rentals 
use_df1=data.drop((["instant","dteday","casual","registered"]), axis=1)
use_df1.head()
#mapping the categorical variables for better understanding
use_df1[['season']] = use_df1[['season']].apply(lambda x: x.map({1:'spring',2:'summer', 3:'fall', 4:'winter'}))
use_df1[['mnth']] = use_df1[['mnth']].apply(lambda x: x.map({1:'Januaray',2:'February', 3:'March', 4:'April', 5:'May', 6:'June', 7:'July', 8:'August', 9:'September', 10:'October', 11:'November',12:'December'}))
use_df1[['weekday']] = use_df1[['weekday']].apply(lambda x: x.map({6:'Saturday',0:'Sunday',1:'Monday', 2:'Tuesday', 3:'Wednesday', 4:'Thursday', 5:'Friday' }))
use_df1[['weathersit']] = use_df1[['weathersit']].apply(lambda x: x.map({ 1:'Clear',2:'Mist', 3:'Light Rain ', 4:'Heavy Rain'}))
use_df1.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.pairplot(use_df1)
plt.show()
# visualizing categorical variables
plt.figure(figsize=(20, 25))
plt.subplot(2,2,1)
sns.boxplot(x = 'season', y = 'cnt', data = use_df1)
plt.subplot(2,2,2)
sns.boxplot(x = 'mnth', y = 'cnt', data = use_df1)
plt.subplot(2,2,3)
sns.boxplot(x = 'weekday', y = 'cnt', data = use_df1)
plt.subplot(2,2,4)
sns.boxplot(x = 'weathersit', y = 'cnt', data = use_df1)
plt.show()
# Get the dummy variables for the feature 'season' and store it in a new variable - season
season = pd.get_dummies(use_df1['season'], drop_first=True)
season.head()
# Get the dummy variables for the feature 'mnth' and store it in a new variable - mnth
mnth = pd.get_dummies(use_df1['mnth'], drop_first=True)
mnth.head()
# Get the dummy variables for the feature 'weekday' and store it in a new variable - weekday
weekday = pd.get_dummies(use_df1['weekday'], drop_first=True)
weekday.head()
# Get the dummy variables for the feature 'weathersit' and store it in a new variable - weathersit
weathersit = pd.get_dummies(use_df1['weathersit'], drop_first=True)
weathersit.head()
#concat dunny varaibles with the dataframe
use_df1 = pd.concat([season,mnth,weekday,weathersit,use_df1], axis= 1)
use_df1.head()
# Drop varaibles for which we have created the dummies for building model
use_df1.drop(['season','mnth','weekday','weathersit'],axis=1, inplace=True)
use_df1
#ploting the correlations of the variables
plt.figure(figsize= (20,15))
sns.heatmap(use_df1.corr(), annot = True, cmap = "GnBu")
plt.show()
#Dropping one of 'atemp' and 'temp' 

use_df1.drop(['temp'], axis = 1, inplace=True)

use_df1.head()

#importing train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split 
#splitting the data set into train and test set in 80-20 ratio
df_train, df_test = train_test_split(use_df1, train_size = 0.8, random_state = 100)
df_train.shape
df_test.shape
#We will use MinMaxScaler to rescale the variables
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'Binary' and 'dummy' variables
num_vars = ['atemp', 'hum', 'windspeed']
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
#We eleminate cnt as we are going to predict it using the model
y_train = df_train.pop("cnt")
X_train = df_train
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
# running RFE
rfe = RFE(lm,15)
rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
column = X_train.columns[rfe.support_]
column
X_train.columns[~rfe.support_]
import statsmodels.api as sm
# Creating X_test dataframe with RFE selected variables
X_train_lm = X_train[column]
#Adding a constant variable
X_train_lm = sm.add_constant(X_train_lm)
# Running the linear model
lm = sm.OLS(y_train,X_train_lm).fit()
#Let's see the summary of our linear model
print(lm.summary())
# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X_train_vif = X_train[column]
vif['Features'] = X_train_vif.columns
vif['VIF'] = [variance_inflation_factor(X_train_vif.values, i) for i in range (X_train_vif.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by= 'VIF', ascending = False)
vif
# We drop hum as it has a very high VIF(>5)
X = X_train_vif.drop('hum', 1)
X_train_lm = sm.add_constant(X)

lr_2 = sm.OLS(y_train, X_train_lm).fit()
print(lr_2.summary())
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# We continue dropping holiday as it has p-value greater than 0.05
X = X_train_vif.drop(['holiday','hum'], 1)
X_train_lm = sm.add_constant(X)

lr_3 = sm.OLS(y_train, X_train_lm).fit()
print(lr_3.summary())
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# We continue dropping atemp as it has a very high VIF(>5)
X = X_train_vif.drop(['atemp','holiday','hum'], 1)
X_train_lm = sm.add_constant(X)

lr_4 = sm.OLS(y_train, X_train_lm).fit()
print(lr_4.summary())
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# We continue dropping winter as it has p-value greater than 0.05
X = X_train_vif.drop(['winter','atemp','holiday','hum'], 1)
X_train_lm = sm.add_constant(X)

lr_5 = sm.OLS(y_train, X_train_lm).fit()
print(lr_5.summary())
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
y_train_cnt = lr_5.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 24)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18) 
# Apply scaler() to all the columns except the 'Binary' and 'dummy' variables
num_vars = ['atemp', 'hum', 'windspeed']

df_test[num_vars] = scaler.transform(df_test[num_vars])

df_test.describe()
#Dividing X_test and y_test
y_test = df_test.pop('cnt')
X_test = df_test
# Creating X_test_lm dataframe by dropping variables from X_test_m4

X_test_lm = X_test[column]
X_test_lm.shape
X_test_lm = X_test_lm.drop(['winter','atemp','holiday','hum'], axis = 1)
X_test_lm.head()
# Adding constant variable to test dataframe
X_test_lm = sm.add_constant(X_test_lm)
# Making predictions using the fourth model
y_pred_lm = lr_5.predict(X_test_lm)
y_pred_lm.head()
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred_lm))
# Calculating the r-squared of the test set
r2_score(y_test, y_pred_lm)
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_lm,)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)     
