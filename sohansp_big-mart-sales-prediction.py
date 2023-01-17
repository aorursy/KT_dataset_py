# Import general packages 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Reading data from the csv files

train_data = pd.read_csv("../input/big-mart-sales-prediction/Train.csv")

test_data = pd.read_csv("../input/big-mart-sales-prediction/Test.csv")

train_data.head()
# Checking the lengths of the available data

print(len(train_data))

print(len(test_data))
# Understanding the training data

train_data.info()
# Checking for missing values

train_data.isnull().any()
# Filling the missing values with the median values

train_data.fillna(train_data.median(),inplace=True)
# Checking to see if missing values still exist. Outlet_Size feature still has missing values

train_data.isnull().any()
plt.scatter(train_data.Item_Visibility, train_data.Item_Outlet_Sales)

plt.title('Item Visibility vs Item Outlet Sales')

plt.xlabel('Item Visibility')

plt.xticks(rotation= 90)

plt.ylabel('Item Outlet Sales')

plt.show()
plt.bar(train_data.Outlet_Identifier, train_data.Item_Outlet_Sales)

plt.title('Outlet_Identifier vs Item Outlet Sales')

plt.xlabel('OutletIdentifier')

plt.xticks(rotation= 90)

plt.ylabel('Item Outlet Sales')

plt.show()
plt.bar(train_data.Item_Type, train_data.Item_Outlet_Sales)

plt.title('Item Type vs Item Outlet Sales')

plt.xlabel('Item Type')

plt.xticks(rotation= 90)

plt.ylabel('Item Outlet Sales')

plt.show()
import seaborn as sns



ax = sns.boxplot(x="Item_Type", y="Item_MRP", data=train_data).set_title("Item Type vs Item MRP")

plt.xticks(rotation= 90)
train_data.info()
# Encoding categorical features so that the model can understand

from sklearn.preprocessing import LabelEncoder

labelEncoder = LabelEncoder()



train_data.Item_Identifier = labelEncoder.fit_transform(train_data.Item_Identifier)

train_data.Item_Fat_Content = labelEncoder.fit_transform(train_data.Item_Fat_Content)

train_data.Item_Type = labelEncoder.fit_transform(train_data.Item_Type)

train_data.Outlet_Identifier = labelEncoder.fit_transform(train_data.Outlet_Identifier)

train_data.Outlet_Location_Type = labelEncoder.fit_transform(train_data.Outlet_Location_Type)

train_data.Outlet_Type = labelEncoder.fit_transform(train_data.Outlet_Type)
# Checking the values of Outlet_Size. 

train_data.Outlet_Size.value_counts()
# How many missing values are there in Outlet_Size. Looks like it has the second heights count. 

train_data.Outlet_Size.isna().sum()
# Correlation matrix for feature selection

corr = train_data.corr()

corr.style.background_gradient(cmap='coolwarm')
# Separating the target variables from all other features. 

# I have removed OutLet_size feature because it has a lot of missing values.

y = train_data.Item_Outlet_Sales

X = train_data.drop(["Item_Outlet_Sales", "Outlet_Size"], axis=1)

X.columns.size
# For confirming feature selection

from sklearn.linear_model import LassoCV



reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)
# Picks out 5 out of 10 features

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
# This graph is not showing the negativity correlated features ("Outlet_Location_Type","Outlet_Type")



imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")
# Considering only the features that have some correlation with the target variables

X = X[["Item_Visibility","Item_MRP","Outlet_Identifier","Outlet_Location_Type","Outlet_Type"]]

X.Item_Visibility.replace({0 : X.Item_Visibility.median()}, inplace = True)
# Spliting the data for the model

from sklearn.model_selection import train_test_split



x_train,x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=42)
# Random Forest Regression model



from sklearn.ensemble import RandomForestRegressor



# Instantiate model with 1000 decision trees

random_forest = RandomForestRegressor(n_estimators = 1000, random_state = 42)



# Train the model on training data

random_forest.fit(x_train, y_train)
from sklearn.metrics import mean_squared_error as mse

from math import sqrt



# Use the forest's predict method on the test data

rf_predictions = random_forest.predict(x_test)



# Print out the root mean square error (RMSE)

rf_rmse = sqrt(mse(y_test, rf_predictions))



print('Root Mean Square Error:', rf_rmse)
plt.plot(y_test, label='Actual')

plt.plot(rf_predictions, label='Predicted')

plt.title("Random Forest")

plt.legend(frameon=True)

plt.show()
# Linear Regression Model



from sklearn import linear_model



linear_regression = linear_model.LinearRegression()

linear_regression.fit(x_train,y_train)
lr_predictions = linear_regression.predict(x_test)



# Print out the root mean square error (RMSE)

lr_rmse = sqrt(mse(y_test, lr_predictions))



print('Root Mean Square Error:', lr_rmse)
plt.plot(y_test, label='Actual')

plt.plot(lr_predictions, label='Predicted')

plt.title("Linear Regression")

plt.legend(frameon=True)

plt.show()
# Decision Tree Regression Model



from sklearn.tree import DecisionTreeRegressor  

  

# create a regressor object 

decision_tree = DecisionTreeRegressor(random_state = 0)  

  

# fit the regressor with X and Y data 

decision_tree.fit(x_train, y_train)
dt_predictions = decision_tree.predict(x_test)



# Print out the root mean square error (RMSE)

dt_rmse = sqrt(mse(y_test, dt_predictions))



print('Root Mean Square Error:', dt_rmse)
plt.plot(y_test, label='Actual')

plt.plot(dt_predictions, label='Predicted')

plt.title("Decision Tree")

plt.legend(frameon=True)

plt.show()
print("Root Mean Sqaure Error of different Regression models:")

print("Random Forest:", rf_rmse)

print("Linear Regression:", lr_rmse)

print("Decision Tree:", dt_rmse)
# Preparing test dataset



test_data = test_data[["Item_Visibility","Item_MRP","Outlet_Identifier","Outlet_Location_Type","Outlet_Type"]]

test_data.Item_Visibility.replace({0 : test_data.Item_Visibility.median()}, inplace = True)

test_data.Outlet_Identifier = labelEncoder.fit_transform(test_data.Outlet_Identifier)

test_data.Outlet_Location_Type = labelEncoder.fit_transform(test_data.Outlet_Location_Type)

test_data.Outlet_Type = labelEncoder.fit_transform(test_data.Outlet_Type)

test_data.head()
# Prediction on test data set

sol = random_forest.predict(test_data)

sol[:10]
solution = pd.read_csv("../input/big-mart-sales-prediction/Submission.csv")

solution.head()
solution['Item_Outlet_Sales'] = sol
# Saving in csv file

solution.to_csv("Submission.csv")