# Importing Libraries

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns
# Importing USA Housing.csv

data = pd.read_csv('../input/USA_Housing.csv')
data.head()
# Checking for Null Values

data.info()
# Getting the summary of Data

data.describe()
# Dropping Address Column

data.drop(['Address'],axis=1,inplace=True)
data.head()
# Let's plot a pair plot of all variables in our dataframe

sns.pairplot(data)
# Visualise the relationship between the features and the response using scatterplots

sns.pairplot(data, x_vars=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population'], y_vars='Price',height=7, aspect=0.7, kind='scatter')
sns.heatmap(data.corr(),annot=True)
data.corr().Price.sort_values(ascending=False)
sns.distplot(data.Price)
from sklearn import preprocessing

pre_process = preprocessing.StandardScaler()
# Putting feature variable to X

X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]



# Putting response variable to y

y = data['Price']
X = pd.DataFrame(pre_process.fit_transform(X))
X.head()
y.head()
#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Importing RFE and LinearRegression

from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lr(Creating LinearRegression Object)

lm = LinearRegression()
# fit the model to the training data

lm.fit(X_train, y_train)
# print the intercept

print(lm.intercept_)
# Let's see the coefficient

coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])

coeff_df
# Making predictions using the model

y_pred = lm.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
from math import sqrt



rms = sqrt(mse)

rms
# Actual and Predicted

c = [i for i in range(1,1501,1)] # generating index 

fig = plt.figure(figsize=(12,8))

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Housing Price', fontsize=16)                       # Y-label
import statsmodels.api as sm

X_train_sm = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X_train_sm = sm.add_constant(X_train_sm)

# create a fitted model in one line

lm_1 = sm.OLS(y_train,X_train_sm).fit()



# print the coefficients

lm_1.params
print(lm_1.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vifs=[vif(X_train.values,i) for i in range(len(X_train.columns))]

pd.Series(data=vifs,index=X_train.columns)
X.head()
X.drop([3],axis=1, inplace=True)
X.head()
#random_state is the seed used by the random number generator, it can be any integer.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# Importing RFE and LinearRegression

from sklearn.linear_model import LinearRegression
# Representing LinearRegression as lr(Creating LinearRegression Object)

lm = LinearRegression()
# fit the model to the training data

lm.fit(X_train, y_train)
# print the intercept

print(lm.intercept_)
# Let's see the coefficient

coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])

coeff_df
# Making predictions using the model

y_pred = lm.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)

print('r_square_value :',r_squared)
from math import sqrt



rms = sqrt(mse)

rms
# Actual and Predicted

c = [i for i in range(1,1501,1)] # generating index 

fig = plt.figure(figsize=(12,8))

plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual

plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted

fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 

plt.xlabel('Index', fontsize=18)                               # X-label

plt.ylabel('Housing Price', fontsize=16)                       # Y-label
import statsmodels.api as sm

X_train_sm = X_train

#Unlike SKLearn, statsmodels don't automatically fit a constant, 

#so you need to use the method sm.add_constant(X) in order to add a constant. 

X_train_sm = sm.add_constant(X_train_sm)

# create a fitted model in one line

lm_1 = sm.OLS(y_train,X_train_sm).fit()



# print the coefficients

lm_1.params
print(lm_1.summary())
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

vifs=[vif(X_train.values,i) for i in range(len(X_train.columns))]

pd.Series(data=vifs,index=X_train.columns)