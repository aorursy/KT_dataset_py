import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Read the data
data = pd.read_csv("../input/insurance.csv")

# See how top 5 rows of the data look like.
data.head()
# How bottom 5 rows look like.
data.tail() 
# Generate statistical summary of the data's numerical features
data.describe()
# View all column names and their respective data types
data.info()
# Check for missing values
print(data.isnull().sum())

#All zeros show that there is no missing value
#-------------------- DATA VISUALIZATION -------------------------
# Visualize distribution of values for target variable - 'charges'
plt.figure(figsize=(6,6))
plt.hist(data.charges, bins = 'auto', color = 'purple')
plt.xlabel("charges ->")
plt.title("Distribution of charges values :")
# Generate Box-plots to check for outliers and relation of each feature with 'charges'
cols = ['age', 'children', 'sex', 'smoker', 'region']
for col in cols:
    plt.figure(figsize=(8,8))
    sns.boxplot(x = data[col], y = data['charges'])
# Converting categorical features' string values to int
# Updating directly to binary because only two values exist
data.smoker = [1 if x == 'yes' else 0 for x in data.smoker]
data.sex = [1 if x == 'male' else 0 for x in data.sex]

# Use pandas because multiple values exist for these columns.
data.region = pd.get_dummies(data.region)
data.charges = pd.to_numeric(data.charges)
data.columns.values
# Create Correlation matrix for all features of data.
data.corr()
# Generate heatmap to visualize strong & weak correlations.
sns.heatmap(data.corr())
# Generate pairplots for all features because there are only 7 in all.
sns.pairplot(data)
#------------------- Prepare data for predictive regression models ----------------------------
y = data.charges.values
X = data.drop(['charges'], axis = 1)   # Drop the target variable
# import scikit learn's built-in Machine learning libraries and functions
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Split using 20% for testing and 80% for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = None)

# ----------------- PREDICTIVE MODELLING (Call the models to be used) -----------------------
rf_reg = RandomForestRegressor(max_features = 'auto', bootstrap = True, random_state = None)
lin_reg = LinearRegression(normalize = True)
ada_reg = AdaBoostRegressor()

# R2-score is used here as a metric. Any other metric could be used instead by just importing 
# it from sklearn

# Predict using Random Forest Regressor.
rf_reg.fit(X_train, y_train)
predtrainRF = rf_reg.predict(X_train)     # Prediction for train data
predtestRF = rf_reg.predict(X_test)       # Prediction for test data

# Compute R-squared score for both train and test data.
print("R2-score on train data:", r2_score(y_train,predtrainRF))
print("R2-score on test data:", r2_score(y_test, predtestRF))

# Predict using Linear Regression
lin_reg.fit(X_train, y_train)
predtrainL = lin_reg.predict(X_train)
predtestL = lin_reg.predict(X_test)
print("R2-score on train data:",r2_score(y_train, predtrainL))
print("R2-score on test data:",r2_score(y_test, predtestL))

# Predict using XGBoost Regressor
ada_reg.fit(X_train, y_train)
predtrainAda = ada_reg.predict(X_train)
predtestAda = ada_reg.predict(X_test)
print("R2-score on train data:",r2_score(y_train, predtrainAda))
print("R2-score on test data:",r2_score(y_test, predtestAda))

# ----------------- Using Ordinary Least Square from Statsmodel --------------------------------
# -------- Allows to view full summary statistics along with p-value and F-statistics -----------
# On Train data.
X_newtrain = sm.add_constant(X_train)
ols_train = sm.OLS(y_train, X_newtrain)
ols_train_new = ols_train.fit()
print(ols_train_new.summary())

# On Test data.
X_newtest = sm.add_constant(X_test)
ols_test = sm.OLS(y_test, X_newtest)
ols_test_new = ols_test.fit()
print(ols_test_new.summary())   # Produce full statistical summary 

plt.show()