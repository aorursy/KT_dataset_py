import sklearn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read the california housing dataset



housing_data = pd.read_csv("/kaggle/input/california-housing-prices/housing.csv")
# See data



housing_data.head()
# See sample view of data for different type of values for categorical features



housing_data.sample(6)
# Describe the dataset



housing_data.describe()
# Check the shape of data



housing_data.shape
# Drop missing records and check the shape



housing_data = housing_data.dropna()

housing_data.shape
# Again use describe dataset



housing_data.describe()
# Check for non-numerical column once



housing_data['ocean_proximity'].unique()
# Total rooms V/S Median house value



fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(housing_data['total_rooms'], housing_data['median_house_value'])

plt.xlabel('Total Rooms')

plt.ylabel('Median House Value')
# Total median income V/S Median house value



fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(housing_data['median_income'], housing_data['median_house_value'])

plt.xlabel('Median Income')

plt.ylabel('Median House Value')
# Check for corealtion with each features



housing_data_corr = housing_data.corr()

housing_data_corr
# Visualize the corelation with help of heatmap



fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(housing_data_corr, annot=True)
# We saw that at 500000 median house value there were some upper cap. Lets count those value



housing_data.loc[housing_data['median_house_value'] >= 500000].count()
# Drop these records



housing_data = housing_data.drop(housing_data.loc[housing_data['median_house_value'] >= 500000].index)
# check the shape



housing_data.shape
housing_data.head()
# Convert categorical col into numerical value



housing_data = pd.get_dummies(housing_data, columns=['ocean_proximity'])
housing_data.shape
housing_data.sample(5)
X = housing_data.drop('median_house_value', axis=1)

Y = housing_data['median_house_value']

X.columns
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)



print(x_train.shape, x_test.shape)

print(y_train.shape, y_test.shape)
from sklearn.linear_model import LinearRegression



linear_model = LinearRegression(normalize=True).fit(x_train, y_train)

print("Training Score : ", linear_model.score(x_train, y_train))
predictors = x_train.columns

predictors
coef = pd.Series(linear_model.coef_, predictors).sort_values()

print(coef)
# Now we have liner model let pridict now :)



y_pred = linear_model.predict(x_test)
# how our model performed, Create a dataframe and check the y_pred and actual values



df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})

df_pred_actual.head(10)
# the r2_score on pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



print("Testing score : ", r2_score(y_test, y_pred))
# Scatter plot between actual and predicted values



fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(y_test, y_pred)
df_pred_actual_sample = df_pred_actual.sample(100)

df_pred_actual_sample = df_pred_actual_sample.reset_index()

df_pred_actual_sample.head()
fig, ax = plt.subplots(figsize=(12, 8))

plt.plot(df_pred_actual_sample['predicted'], label='Predicted')

plt.plot(df_pred_actual_sample['actual'], label='Actual')

plt.ylabel('Median House Value')

plt.legend()

plt.show()
