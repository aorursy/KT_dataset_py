# Importing the libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Importing the dataset

df = pd.read_csv('../input/BlackFriday.csv')
df.head()
# Replacing NaN from columns "Product_Category_2" and "Product_Category_3"

df['Product_Category_2'].fillna(0, inplace=True)

df['Product_Category_3'].fillna(0, inplace=True)
# Dropping the fields 'User_ID' and 'Product_ID'

df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)

df.head()
df.info()
# Marital deployment

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(df['Marital_Status'].value_counts(), explode = (0.2,0), labels=['Single', 'Married'], autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()

plt.title('Marital status deployment')
# Gender status deployment

fig1, ax1 = plt.subplots(figsize=(10,5))

ax1.pie(df['Gender'].value_counts(), explode = (0.2,0), labels=['Male', 'Female'], autopct='%1.1f%%',

        shadow=True, startangle=90)

plt.legend()

plt.title('Gender deployment')
# Gender groups with marritial status

plt.figure(figsize=(10,6))

sns.countplot(df['Gender'], hue=df['Marital_Status'])
# Age groups with gender deployment

plt.figure(figsize=(10,6))

sns.countplot(df['Age'].sort_values(), hue=df['Gender'])
# City category with age deployment

plt.figure(figsize=(10,6))

sns.countplot(df['City_Category'].sort_values(), hue=df['Age'].sort_values())
df.dtypes
# Taking care of categorical data

df.Age.replace({'0-17':0,

                '18-25':1, 

                '26-35':2,

                '36-45':3,

                '46-50':4,

                '51-55':5,

                '55+':6}, inplace = True)

df.City_Category.replace({'A':1,

                          'B':2,        

                          'C':3}, inplace = True)

df.Gender.replace({'M':1,        

                   'F':0}, inplace = True)

df.Stay_In_Current_City_Years.replace({'0':0,

                                       '1':1,        

                                       '2':2,        

                                       '3':3,        

                                       '4+':4}, inplace = True)



df.head()
# Creating DV and IV sets

X = df.drop('Purchase', axis=1)

y = df['Purchase']



# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=123)
"""

We are goinig to test 4 models:

- Simple Linear Regression

- Decision Tree Regression

- Random Forest Regression

- Gradient Boosting Regression



For evaluation we have 4 metrics: Mean Square Error, R2, Mean accuracy (from k-Fold Cross Validation), Standard deviation (from k-Fold Cross Validation).

All metrics are to be stored in 'sum_met' data frame (summary of meterics).

"""

# Metrics of used Regression models

sum_met = pd.DataFrame(index = ["RMSE_Error", "R2_Score", "Mean_accuracy", "Std_deviation"],

                       columns = ['Lin_Reg', 'Decision_Tree', 'Random_Forest', 'G_Boost'])
# Simple Linear Regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Finding the mean_squared error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)



# Finding the r2 score or the variance (R2)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10)



# Printing metrics for Linear Regression

print("RMSE Error:", np.sqrt(mse))

print("R2 Score:", r2)

print("Mean accuracy:", accuracies.mean())

print("Std deviation:", accuracies.std())



# Writing metrics to summary data frame

sum_met.at['RMSE_Error','Lin_Reg'] = np.sqrt(mse)

sum_met.at['R2_Score','Lin_Reg'] = r2

sum_met.at['Mean_accuracy','Lin_Reg'] = accuracies.mean()

sum_met.at['Std_deviation','Lin_Reg'] = accuracies.std()
# Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 123)

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Finding the mean_squared error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)



# Finding the r2 score or the variance (R2)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10)



# Printing metrics

print("RMSE Error:", np.sqrt(mse))

print("R2 Score:", r2)

print("Mean accuracy:", accuracies.mean())

print("Std deviation:", accuracies.std())



# Writing metrics to summary data frame

sum_met.at['RMSE_Error','Decision_Tree'] = np.sqrt(mse)

sum_met.at['R2_Score','Decision_Tree'] = r2

sum_met.at['Mean_accuracy','Decision_Tree'] = accuracies.mean()

sum_met.at['Std_deviation','Decision_Tree'] = accuracies.std()
# Random Forest Regression

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 300, random_state = 123) 

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Finding the mean_squared error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)



# Finding the r2 score or the variance (R2)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10)



# Printing metrics

print("RMSE Error:", np.sqrt(mse))

print("R2 Score:", r2)

print("Mean accuracy:", accuracies.mean())

print("Std deviation:", accuracies.std())



# Writing metrics to summary data frame

sum_met.at['RMSE_Error','Random_Forest'] = np.sqrt(mse)

sum_met.at['R2_Score','Random_Forest'] = r2

sum_met.at['Mean_accuracy','Random_Forest'] = accuracies.mean()

sum_met.at['Std_deviation','Random_Forest'] = accuracies.std()
# Gradient Boosting Regression

from sklearn.ensemble import GradientBoostingRegressor

regressor = GradientBoostingRegressor(n_estimators = 100, max_depth = 5, min_samples_split = 2, learning_rate = 0.1)

regressor.fit(X_train, y_train)



# Predicting the Test set results

y_pred = regressor.predict(X_test)



# Finding the mean_squared error (MSE)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)



# Finding the r2 score or the variance (R2)

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)



# Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_test, y = y_test, cv = 10)



# Printing metrics

print("RMSE Error:", np.sqrt(mse))

print("R2 Score:", r2)

print("Mean accuracy:", accuracies.mean())

print("Std deviation:", accuracies.std())



# Writing metrics to summary data frame

sum_met.at['RMSE_Error','G_Boost'] = np.sqrt(mse)

sum_met.at['R2_Score','G_Boost'] = r2

sum_met.at['Mean_accuracy','G_Boost'] = accuracies.mean()

sum_met.at['Std_deviation','G_Boost'] = accuracies.std()
sum_met