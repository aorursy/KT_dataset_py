# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
# Import the necessary packages and libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_columns', 50)



from sklearn.model_selection import train_test_split



plt.style.use('ggplot')
# Read the data



bike = pd.read_csv('/kaggle/input/boombikes/day.csv')

bike.head()
# Checking the shape of the dataframe



bike.shape
# Checking the info for all the columns



bike.info()
# Dropping the unnecessary columns



bike = bike.drop(['instant','dteday','casual','registered','atemp'], axis=1)

bike.shape
# Renaming the columns for better understanding



bike.rename(columns = {'yr':'year','mnth':'month','hum':'humidity','cnt':'count'}, inplace = True) 

bike.head()
bike.nunique().sort_values()
# Mapping variables season, month, weekday, weathersit



bike['season'] = bike.season.map({1: 'spring', 2: 'summer',3:'fall', 4:'winter'})



bike['month'] = bike.month.map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'June', 7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'})



bike['weekday']  =bike.weekday.map({0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu',  5:'Fri', 6:'Sat'})



bike['weathersit'] = bike.weathersit.map({1: 'Clear', 2:'Cloudy', 3:'Light Snow', 4:'Heavy Rain'})



bike.head()
# Pairplot of all the numeric variables



sns.pairplot(bike, vars=['count', 'temp', 'humidity', 'windspeed'])

plt.show()
#Boxplot for some of the categorical variables with respect to the target varibale 'count'



plt.figure(figsize=(20, 12))

plt.subplot(2,4,1)

sns.boxplot(x = 'year', y = 'count', data = bike)

plt.subplot(2,4,2)

sns.boxplot(x = 'holiday', y = 'count', data = bike)

plt.subplot(2,4,3)

sns.boxplot(x = 'workingday', y = 'count', data = bike)

plt.subplot(2,4,4)

sns.boxplot(x = 'weathersit', y = 'count', data = bike)

plt.subplot(2,4,5)

sns.boxplot(x = 'season', y = 'count', data = bike)

plt.subplot(2,4,6)

sns.boxplot(x = 'weekday', y = 'count', data = bike)

plt.subplot(2,4,7)

sns.boxplot(x = 'month', y = 'count', data = bike)

plt.show()
# Analysis between weathersit and count



plt.figure(figsize=(10,4))

sns.barplot('weathersit','count',data=bike)

plt.title('Bike Rentals in different Weather Situations',fontsize=12)

plt.show()
# Analysis between season and count



plt.figure(figsize=(10,4))

sns.barplot('season','count',data=bike)

plt.title('Bike Rentals in different Seasons',fontsize=12)

plt.show()
# Analysis of the Bike Rentals for each month of both the years



plt.figure(figsize=(10,5))

sns.barplot('month','count',hue='year',data=bike)

plt.title('Bike Rentals in different Months of both the Years',fontsize=12)

plt.show()
# Analysis of Bike Rentals with Temperature



sns.scatterplot(x='temp',y='count' ,data=bike)

plt.title('Temp vs Count')

plt.show()
# Analysis of Bike Rentals with Humidity



sns.scatterplot(x='humidity',y='count' ,data=bike)

plt.title('Humidity vs Count')

plt.show()
# Heatmap to visualise the correlation between the variables



plt.figure(figsize=(10, 5))

sns.heatmap(bike.corr(), cmap="YlGnBu", annot = True)

plt.title("Correlation between the variables")

plt.show()
month_dummy = pd.get_dummies(bike.month,drop_first=True)

weekday_dummy = pd.get_dummies(bike.weekday,drop_first=True)

weathersit_dummy = pd.get_dummies(bike.weathersit,drop_first=True)

season_dummy = pd.get_dummies(bike.season,drop_first=True)
# Adding the dummy variables to the original dataframe



bike = pd.concat([month_dummy,weekday_dummy,weathersit_dummy,season_dummy,bike],axis=1)

bike.head()
# Dropping the original columns - month, weekday, weathersit and season, since dummy variables have already been created for them



bike.drop(['season','month','weekday','weathersit'], axis = 1, inplace = True)

bike.shape
np.random.seed(0)



bike_train, bike_test = train_test_split(bike, train_size = 0.7, random_state = 100)



print(bike_train.shape)

print(bike_test.shape)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Creating a list of numeric variables



num_vars=['temp','humidity','windspeed','count']
# Fit the data



bike_train[num_vars] = scaler.fit_transform(bike_train[num_vars])

bike_train.head()
# Checking numeric variables after scaling the features

bike_train.describe()
y_train = bike_train.pop('count')

X_train = bike_train
# Importing RFE and LinearRegression



from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 15



lm = LinearRegression()

lm.fit(X_train, y_train)



# Running RFE

rfe = RFE(lm, 15)             

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
# Columns for which rfe_support is true



col = X_train.columns[rfe.support_]

col
# Creating the first dataframe model with RFE selected variables

X_train_1 = X_train[col]
# Adding a constant variable 



import statsmodels.api as sm  

X_train_1 = sm.add_constant(X_train_1)
# Running the linear model



lm = sm.OLS(y_train,X_train_1).fit() 
# Summary of our linear model

print(lm.summary())
# Dropping the const variable



X_train_1 = X_train_1.drop(['const'], axis=1)
# Calculating the VIFs for the new model



from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_1

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'Sat'



X_train_2 = X_train_1.drop(['Sat'], axis=1)
# Adding the contsant variable



X_train_2 = sm.add_constant(X_train_2)
# Running the linear model



lm = sm.OLS(y_train,X_train_2).fit() 
# Summary of the new model

print(lm.summary())
# Dropping the const variable



X_train_2 = X_train_2.drop(['const'], axis=1)
# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_2

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'humidity'



X_train_3 = X_train_2.drop(['humidity'], axis=1)
# Adding the contsant variable



X_train_3 = sm.add_constant(X_train_3)
# Running the linear model



lm = sm.OLS(y_train,X_train_3).fit()
# Summary of the new model

print(lm.summary())
# Dropping the const variable



X_train_3 = X_train_3.drop(['const'], axis=1)
# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_3

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'windspeed'



X_train_4 = X_train_3.drop(['windspeed'], axis=1)
# Adding the contsant variable



X_train_4 = sm.add_constant(X_train_4)
# Running the linear model



lm = sm.OLS(y_train,X_train_4).fit()
# Summary of the new model

print(lm.summary())
# Dropping the const variable



X_train_4 = X_train_4.drop(['const'], axis=1)
# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_4

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'Jan'



X_train_5 = X_train_4.drop(['Jan'], axis=1)
# Adding the contsant variable



X_train_5 = sm.add_constant(X_train_5)
# Running the linear model



lm = sm.OLS(y_train,X_train_5).fit()
# Summary of the new model

print(lm.summary())
# Dropping the const variable



X_train_5 = X_train_5.drop(['const'], axis=1)
# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_5

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'Dec'



X_train_6 = X_train_5.drop(['Dec'], axis=1)
# Adding the contsant variable



X_train_6 = sm.add_constant(X_train_6)
# Running the linear model



lm = sm.OLS(y_train,X_train_6).fit()
# Summary of the new model

print(lm.summary())
# Dropping the const variable



X_train_6 = X_train_6.drop(['const'], axis=1)

# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_6

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Rebuilding the model without 'Nov'



X_train_7 = X_train_6.drop(['Nov'], axis=1)
# Adding the contsant variable



X_train_7 = sm.add_constant(X_train_7)
# Running the linear model



lm = sm.OLS(y_train,X_train_7).fit()
# Summary of the new model

print(lm.summary())
# Calculating the VIFs for the new model



vif = pd.DataFrame()

X = X_train_7

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif = vif[vif['Features']!='const']    # Ignoring to display the vif of 'const'

vif
lm = sm.OLS(y_train,X_train_7).fit()  #As obtained previously

y_train_count = lm.predict(X_train_7)
# Plot the histogram of the error terms



fig = plt.figure()

sns.distplot((y_train - y_train_count), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label

plt.show()
num_vars=['temp','humidity','windspeed','count']



# Fit and transform operations are done on the training data but only transform operation will be done on the test data



bike_test[num_vars] = scaler.transform(bike_test[num_vars])

bike_test.describe()
y_test = bike_test.pop('count')

X_test = bike_test
# Adding constant variable to test dataframe



X_test_m7 = sm.add_constant(X_test)
# Creating X_test_m7 dataframe by dropping variables which were removed till our final Model 7 in the training dataset



X_test_m7 = X_test_m7.drop(['Aug','Feb','June','Mar','May','Oct','Mon','Sun','Thu','Tue','Wed','summer','workingday','Sat','humidity','windspeed','Jan','Dec','Nov'], axis = 1)
# Making predictions using the seventh model



y_pred_m7 = lm.predict(X_test_m7)
# Plotting y_test and y_pred to understand the spread



fig = plt.figure()

plt.scatter(y_test, y_pred_m7)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)      

plt.show()
# Regression plot



sns.regplot(x = y_test, y = y_pred_m7, fit_reg=True,scatter_kws={"color": "blue"}, line_kws={"color": "red"})



plt.title('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label

plt.show()
# Evaluate R-square for test dataset



from sklearn.metrics import r2_score

r2_score(y_test,y_pred_m7)
# Adjusted R^2

# adj r2 = 1-((1-R2)*(n-1)/(n-p-1))



# n = sample size (in this case the value is 220, as yielded before)

# p = number of independent variables(in this case the value is 9)



Adj_r2 = 1 - ((1 - 0.8065842474886509) * 219 / (220-9-1))

print(Adj_r2)