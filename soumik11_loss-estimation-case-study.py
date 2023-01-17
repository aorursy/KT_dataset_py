# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Importing the libraries 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
my_data=pd.read_csv("../input/case-study-loss-given/R_Module_Day_5.2_Data_Case_Study_Loss_Given_Default.csv") 

my_data.head()
# Check for missing values

my_data.isnull().sum()
# See rows with missing values

my_data[my_data.isnull().any(axis=1)]
my_data.describe()
## Seaborn visualization library

sns.set(style="ticks", color_codes=True)

## Create the default library

sns.pairplot(my_data)
# RuntimeWarning: invalid value encountered in divide <to avoid this I am using np.seterr>

np.seterr(divide='ignore', invalid='ignore')

sns.pairplot(my_data, hue= 'Age')
# RuntimeWarning: invalid value encountered in divide <to avoid this I am using np.seterr>

np.seterr(divide='ignore', invalid='ignore')

sns.pairplot(my_data, hue= 'Years of Experience')
#normal plot using seaborn

sns.lmplot(x="Age",y="Losses in Thousands",data=my_data,size=10)
sns.jointplot(data=my_data, x='Years of Experience', y='Losses in Thousands', kind='hex', color="#4CB391",size=10)
tableau_20=(197/255.,101/255.,213/255.)

sns.jointplot(data=my_data, x='Number of Vehicles', y='Losses in Thousands',space=0.2, kind='kde', color=tableau_20,size=10)
my_data1=my_data.corr().transpose()

my_data1.head()
plt.figure(figsize=(20,15))

sns.heatmap(my_data1)
sns.set()

# Load the example my_data1 dataset and conver to long-form

#my_data_float = my_data1

data_vis = my_data1.pivot("Losses in Thousands","Age","Years of Experience")



# Draw a heatmap with the numeric values in each cell

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(data_vis, annot=True, fmt="f", linewidths=.5, ax=ax)
# Spliting target variable and independent variables

X = my_data1.drop(['Losses in Thousands'], axis = 1)

y = my_data1['Losses in Thousands']
# Splitting to training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 4)
# Import library for Linear Regression

from sklearn.linear_model import LinearRegression



# Create a Linear regressor

lm = LinearRegression()



# Train the model using the training sets 

lm.fit(X_train, y_train)
# Value of y intercept

lm.intercept_
#Converting the coefficient values to a dataframe

coeffcients = pd.DataFrame([X_train.columns,lm.coef_]).T

coeffcients = coeffcients.rename(columns={0: 'Attribute', 1: 'Coefficients'})

coeffcients
# Model prediction on train data

y_pred = lm.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual losses and predicted losses

plt.scatter(y_train, y_pred)

plt.xlabel("Losses")

plt.ylabel("Predicted losses")

plt.title("Losses vs Predicted Losses")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
# Checking Normality of errors

sns.distplot(y_train-y_pred)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
# Predicting Test data with the model

y_test_pred = lm.predict(X_test)
# Model Evaluation

acc_linreg = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_linreg)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
# Import Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor



# Create a Random Forest Regressor

reg = RandomForestRegressor()



# Train the model using the training sets 

reg.fit(X_train, y_train)
# Model prediction on train data

y_pred = reg.predict(X_train)
# Model Evaluation

print('R^2:',metrics.r2_score(y_train, y_pred))

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_train, y_pred))*(len(y_train)-1)/(len(y_train)-X_train.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_train, y_pred))

print('MSE:',metrics.mean_squared_error(y_train, y_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
# Visualizing the differences between actual losses and predicted losses

plt.scatter(y_train, y_pred)

plt.xlabel("Losses")

plt.ylabel("Predicted losses")

plt.title("Losses vs Predicted Losses")

plt.show()
# Checking residuals

plt.scatter(y_pred,y_train-y_pred)

plt.title("Predicted vs residuals")

plt.xlabel("Predicted")

plt.ylabel("Residuals")

plt.show()
# Checking Normality of errors

sns.distplot(y_train-y_pred)

plt.title("Histogram of Residuals")

plt.xlabel("Residuals")

plt.ylabel("Frequency")

plt.show()
# Predicting Test data with the model

y_test_pred = reg.predict(X_test)
# Model Evaluation

acc_rf = metrics.r2_score(y_test, y_test_pred)

print('R^2:', acc_rf)

print('Adjusted R^2:',1 - (1-metrics.r2_score(y_test, y_test_pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

print('MAE:',metrics.mean_absolute_error(y_test, y_test_pred))

print('MSE:',metrics.mean_squared_error(y_test, y_test_pred))

print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))