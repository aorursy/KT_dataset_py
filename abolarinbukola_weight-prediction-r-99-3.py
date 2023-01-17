#importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
path = '/kaggle/input/fish-market/Fish.csv'
df = pd.read_csv(path)
df.head(3)
df.describe(include = 'all')
df.info()
#plotting a distribution plot

sns.distplot(df['Weight']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Weight'].quantile(0.99)

df = df[df['Weight']<q]



sns.distplot(df['Weight']) 
sns.distplot(df['Length1']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Length1'].quantile(0.99)

df = df[df['Length1']<q]



sns.distplot(df['Length1'])
sns.distplot(df['Length2']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Length2'].quantile(0.99)

df = df[df['Length2']<q]



sns.distplot(df['Length2'])
sns.distplot(df['Length3']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Length3'].quantile(0.99)

df = df[df['Length3']<q]



sns.distplot(df['Length3'])
#We need to reset index of the dataframe after droppinh those observation

df.reset_index(drop = True, inplace = True)
df.describe()
sns.distplot(df['Height']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Height'].quantile(0.99)

df = df[df['Height']<q]



sns.distplot(df['Height'])
sns.distplot(df['Width']) #we can see those few outliers shown by the longer right tail of the distribution
#Removing the top 1% of the observation will help us to deal with the outliers

q = df['Width'].quantile(0.99)

df = df[df['Width']<q]



sns.distplot(df['Width'])
df.describe()
df.columns.values
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



# the target column (in this case 'weight') should not be included in variables

#Categorical variables already turned into dummy indicator may or maynot be added if any

variables = df[['Length1', 'Length2', 'Length3', 'Height','Width']]

X = add_constant(variables)

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]

vif['features'] = X.columns

vif
df.drop(['Length1','Length2'], axis = 1, inplace = True)
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant



variables = df[['Length3', 'Height','Width']]

X = add_constant(variables)

vif = pd.DataFrame()

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]

vif['features'] = X.columns

vif
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))

ax1.scatter(df['Length3'], df['Weight'])

ax1.set_title('Length3 and weight')



ax2.scatter(df['Height'], df['Weight'])

ax2.set_title('Height and weight')



ax3.scatter(df['Width'], df['Weight'])

ax3.set_title('Width and weight')
#Creating a new column in our dataset containing log-of-weight

df['log_weight'] = np.log(df['Weight'])
#RE plotting the graphs but this time using 'log_weight' as our target variable

fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))

ax1.scatter(df['Length3'], df['log_weight'])

ax1.set_title('Length3 and log_weight')



ax2.scatter(df['Height'], df['log_weight'])

ax2.set_title('Height and log_weight')



ax3.scatter(df['Width'], df['log_weight'])

ax3.set_title('Width and log_weight')
#Creating new columns to hold the logs of the variables

df['log_length3'] = np.log(df['Length3'])



df['log_width'] = np.log(df['Width'])



df['log_height'] = np.log(df['Height'])
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))

ax1.scatter(df['log_length3'], df['log_weight'])

ax1.set_title('log_weight and log_length 1')

ax2.scatter(df['log_width'], df['log_weight'])

ax2.set_title('log_width and log_weight')

ax3.scatter(df['log_height'], df['log_weight'])

ax3.set_title('log_height and log_weight')
#Getting the present variables in our dataframe

df.columns.values
#Dropping those columns that as been logged

df = df.drop(['Length3', 'Height', 'Width', 'Weight'], axis = 1)
df.describe()
df[df['log_weight'].apply(lambda x: x < 1000)].sort_values('log_weight', ascending = True)
#Drop a row with index number 40

df.drop([40], inplace = True)



#Resetting the index after dropping a row

df.reset_index(drop = True, inplace = True)
df = pd.get_dummies(df, drop_first = True)
df.head()
#Declaring independent variable i.e x

#Declaring Target variable i.e y

y = df['log_weight']

x = df.drop(['log_weight'], axis = 1)
scaler = StandardScaler() #Selecting the standardscaler

scaler.fit(x)#fitting our independent variables
scaled_x = scaler.transform(x)#scaling
#Splitting our data into train and test dataframe

x_train,x_test, y_train, y_test = train_test_split(scaled_x, y , test_size = 0.2, random_state = 47)
reg = LinearRegression()#Selecting our model

reg.fit(x_train,y_train)
#predicting using x_train

y_hat = reg.predict(x_train)
#Plotting y_train vs our predicted value(y_hat)

fig, ax = plt.subplots()

ax.scatter(y_train, y_hat)
#Residual graph

sns.distplot(y_train - y_hat)

plt.title('Residual Graph')
#R2

reg.score(x_train, y_train)
#Intercept of the regression line

reg.intercept_
#Coefficient

reg.coef_
#Predicting with x_test

y_hat_test = reg.predict(x_test)
reg.score(x_test, y_test)
#Plotting predicted value against y_test

plt.scatter(y_test, y_hat_test, alpha=0.5)

plt.show()
#Creating a summary table containing coefficients for each variable

summary = pd.DataFrame( data = x.columns.values, columns = ['Features'] )

summary['Weight'] = reg.coef_

summary
#Creating a new dataframe

df1 = pd.DataFrame( data = np.exp(y_hat_test), columns = ['Predictions'] )
#Resetting index to match the index of y_test with that of the dataframe

y_test = y_test.reset_index(drop = True)
#target column will hold our predicted values

df1['target'] = np.exp(y_test)

#Substrating predictions from target to get the difference in value

df1['Residual'] = df1['target'] - df1['Predictions']



#Difference in percentage

df1['Difference%'] = np.absolute(df1['Residual']/ df1['target'] * 100)
df1.describe()
df1.sort_values('Difference%')