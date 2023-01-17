# Importing filterwarnings to ignore warning messages

import warnings

warnings.filterwarnings('ignore')
# Importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Reading the bikes rental  dataset into 'bikes' dataframe



bikes=pd.read_csv("../input/boom-bikes-data/day.csv")

bikes.head()
# looking at the shape of the bikes dataset

bikes.shape
bikes.describe()
#Finding out the datatype of the columns in the bikes dataset:

bikes.info()
# To check if there are any missing values in the dataset



import missingno as mn

mn.matrix(bikes)
# Dropping the columns : instant,dteday,casual,registered



cols=["instant","dteday","casual","registered"]



bikes=bikes.drop(columns=cols,axis=1)
# Renaming some columns for more clearity 



bikes.rename(columns={'hum':'humidity','cnt':'count','mnth':'month','yr':'year'},inplace=True)
# Mapping the categorical column : season into its categories



season_cat={1:"spring",2:"summer",3:"fall",4:'winter'}



bikes.season=[season_cat[item] for item in bikes.season]
# Mapping the categorical column : weathersit into its categories



weather_cat={1:"clear",2:"mist & cloudy",3:"light rain & snow",4:'heavy rain & snow'}



bikes.weathersit=[weather_cat[item] for item in bikes.weathersit]
# Mapping the categorical column : month into its categories



month_cat={1: 'Jan' , 2: 'Feb' , 3: 'Mar' , 4: 'Apr' , 5: 'May' , 6: 'Jun' , 7: 'Jul' , 8: 'Aug' , 9: 'Sep' , 10: 'Oct' , 11: 'Nov' , 12: 'Dec'}



bikes.month=[month_cat[item] for item in bikes.month]
# Mapping the categorical column : weekday into its categories



wkday_cat={0: 'Sunday',1: 'Monday',2: 'Tuesday',3: 'Wednesday',4: 'Thursday',5: 'Friday',6: 'Saturday'}



bikes.weekday=[wkday_cat[item] for item in bikes.weekday]
# Mapping the categorical column : Year into its categories



yr_cat={0: '2018',1: '2019'}



bikes.year=[yr_cat[item] for item in bikes.year]
# Analysing the demand in various seasons

sns.barplot(x='season',y='count',data=bikes)
# Analysing the demand in year 2018 and 2019

sns.set_style('whitegrid')

plt.figure(figsize=(6,4))

sns.barplot(x='year',y='count',data=bikes)
# Analysing the demand in various months

sns.set_style('whitegrid')

plt.figure(figsize=(12,6))

sns.barplot(x='month',y='count',data=bikes,hue='year',palette='ocean')
# Analysing the demand in various weathers

sns.set_style('whitegrid')

plt.figure(figsize=(9,4))

sns.barplot(x='weathersit',y='count',data=bikes)
# Analysing the demand in various weekdays

sns.set_style('whitegrid')

plt.figure(figsize=(9,4))

sns.barplot(x='weekday',y='count',data=bikes)
# Analysing the demand based on workingday or not a workingday

sns.set_style('whitegrid')

# plt.figure(figsize=(9,4))

sns.barplot(x='workingday',y='count',data=bikes)
sns.pairplot(bikes, x_vars=['temp','atemp','humidity','windspeed'], y_vars='count',size=4, aspect=1 )

plt.show()
plt.figure(figsize=(11,7.5))

sns.heatmap(bikes.corr(),annot=True,cmap='YlGnBu')
# Dropping the variable 'atemp' 

bikes=bikes.drop("atemp",axis=1)
# Creating the dummy variables for the variables month,season,weathersit,weekday and storing them 

# in new variable 'months',seasons','weather' and 'weekdays' respectively and 

# dropping the first column from these variables using 'drop_first = True'



months= pd.get_dummies(bikes['month'],drop_first=True,prefix='month')



seasons = pd.get_dummies(bikes['season'],drop_first=True,prefix='season')



weather= pd.get_dummies(bikes['weathersit'],drop_first=True,prefix='weather')



weekdays= pd.get_dummies(bikes['weekday'],drop_first=True,prefix='day')



years= pd.get_dummies(bikes['year'],drop_first=True,prefix='year')
# Add the above created dummy variables to the original bikes dataframe

bikes = pd.concat([bikes,months,seasons,weather,weekdays,years], axis = 1)



# Looking at the top rows of our dataframe.

bikes.head()
# As we have created dummy variables for the categorical variables , now we will drop those categorical variables .



bikes.drop(['season','weathersit','weekday','month','year'],axis=1,inplace=True)
# Looking at the shape of dataframe after dropping the above variables

bikes.shape
from sklearn.model_selection import train_test_split



bikes_train, bikes_test = train_test_split(bikes, train_size = 0.7, test_size = 0.3, random_state = 100)
#Looking at the shape of the train dataset.

bikes_train.shape
#Looking at the shape of the test dataset.

bikes_test.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Applying Scaling on the continuous columns : 'windspeed' , 'temp' , 'humidity' , 'count'

vars = ['windspeed' , 'temp' , 'humidity','count']



bikes_train[vars] = scaler.fit_transform(bikes_train[vars])



bikes_train.head()
y_train = bikes_train.pop('count')

X_train = bikes_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with selecting 15 variables 

lm = LinearRegression()

lm.fit(X_train, y_train)



np.random.seed(0)

rfe = RFE(lm, 15)             # running RFE,15 is the number of variables we want RFE to select

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))  



# rfe_support_ : tells whether RFE selected the variable or not

# rfe.ranking_ : tells the next best variable to be selected and ranks accordingly , The numbers 

#                 beside the variables indicate the importance of that variable.
# Looking at the cols that RFE selected

col = X_train.columns[rfe.support_]

col
# Creating X_train_rfe dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
# Running the linear model

lm = sm.OLS(y_train,X_train_rfe).fit()   
#Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_rfe

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the const variable

X_train_new = X_train_rfe.drop(["const"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the 'humidity' variable

X_train_new = X_train_new.drop(["humidity"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the 'workingday' variable

X_train_new = X_train_new.drop(["workingday"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Dropping the 'day_Saturday' variable

X_train_new = X_train_new.drop(["day_Saturday"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the 'month_Jan' variable

X_train_new = X_train_new.drop(["month_Jan"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the 'month_Sep' variable

X_train_new = X_train_new.drop(["month_Sep"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Dropping the 'season_summer' variable

X_train_new = X_train_new.drop(["season_summer"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)



#Running the linear model

lm = sm.OLS(y_train,X_train_lm).fit()



##Looking at the summary of our linear model

lm.summary()
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_count = lm.predict(X_train_lm)
# Plotting the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_count), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                   

plt.xlabel('Errors', fontsize = 18)      
# Applying Scaling on the continuous columns : 'windspeed' , 'temp' , 'humidity'

vars = ['windspeed' , 'temp' , 'humidity','count']



bikes_test[vars] = scaler.transform(bikes_test[vars])
y_test = bikes_test.pop('count')

X_test = bikes_test
# Using our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16) 
from sklearn.metrics import r2_score

r2=r2_score(y_test, y_pred)

r2
n = X_test_new.shape[0]      # n is number of rows in X_test_new



p = X_test_new.shape[1]     # p= Number of features/predictors which is number of columns in X_test_new



# Calculating Adjusted R-squared value using the formula



adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)

adjusted_r2
# calculating the Mean Squared Error , Root Mean Squared Error and Mean Absolute error

from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))