import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
# read the data

data=pd.read_csv('../input/50_Startups.csv')



# Look 

data.head() 



# Profit is a target variable (value to be predict)

# the remaining fields are features
# Check, if null value

data.isnull().sum() 



# Great!! there is no missing values. 
data.info()

# the data has four fields of float type -- R&D spend, Adminstration, marketing Spend

# the data has one field of object type -- State
# Lookign for correlation

corr_matrix = data.corr()

corr_matrix["Profit"].sort_values(ascending=False)

# It seems that field "R&D Spend" and "Marketing Spend" have a strong correlation. 

# Let's look it more closely these relationship
# Corrleation between target variable ("Profit") and "R&D Spend"

%matplotlib inline

data.plot(kind="scatter", x="R&D Spend", y="Profit", alpha=1.0)



# Early observation -- It seems that there is a linear relationship between R&D speed and Profile
# Corrleation between target variable ("Profit") and "R&D Spend"

%matplotlib inline

data.plot(kind="scatter", x="Marketing Spend", y="Profit", alpha=1.0) 



# Early observation -- It seems that there is a linear relationship between Marketing speed and Profile, 

# but not as strong relationship as "R&D spend "
# Corrleation between target variable ("Profit") and "Administration     "

%matplotlib inline

data.plot(kind="scatter", x="Administration", y="Profit", alpha=1.0) 



# Early observation -- 

# It seems that there is no  linear relationship between Administration  and Profile, as the graph is scatter

# "Adminstration" field may be the candidate of removal
# Let's try to identify the relationship between "State" and "SalePrice"



data.groupby(["State"]).mean()["Profit"].sort_values(ascending=False)



# Observations 

# It seems that there is a relation between "State" and Profit

# Florida records highest average profit 

# California records lowest  average profit 
# Let's first separate "target variable" (i.e., profile) and feature variables.

# We drop the Profit from dataset

y = data["Profit"]

X = data.drop(['Profit'],axis=1)
# Let's encode "State" Variable

# State is only categorical variable with 3 labels.

X.State.value_counts()
# Because of 3 labels I am using on hot encoder.  

X=pd.get_dummies(X)

X.head()
# Let's first split the data into "Training" and "Testing" dataset

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#  Linear Regression model

# Train the Linear Regression model

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
# Predict

predictions=regressor.predict(X_test)
# find the performance of Linear Regression  -- RMSE (Root Mean Square Error)

from sklearn.metrics import mean_squared_error

from math import sqrt



mse=round((mean_squared_error(y_test,predictions))/100, 2)

rmse = round((sqrt(mse))/100 ,2)

print ("MSE", mse)

print ("RMSE", rmse)
import statsmodels.api as sm



X=sm.add_constant(X)

# Add "1" const as one column.

# This is for multivariable regression constant - C

# y  = C + M1X1 + M2X2 + M3X3 + ...  MnXn



X.head()
model=sm.OLS(y,X).fit()

model.summary()
X=X.drop(['Administration'],axis=1)

model=sm.OLS(y,X).fit()

model.summary()
X=X.drop(['Marketing Spend'],axis=1)

model=sm.OLS(y,X).fit()

model.summary()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X_train,y_train)
predictions =regressor.predict(X_test)
mse=round((mean_squared_error(y_test,predictions))/100, 2)

rmse = round((sqrt(mse))/100 ,2)

print ("MSE", mse)

print ("RMSE", rmse)