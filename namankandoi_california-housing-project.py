# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# set filepath for reading
filepath = "../input/simplilearn-projects/Data California Housing.xlsx"
# import dataset into notebook
housing_data = pd.read_excel(filepath)
# check dataframe format
housing_data
# check which columns have NaN values
housing_data.info()
# confirm number of NaN values
housing_data["total_bedrooms"].isnull().value_counts()
# get average value to replace NaN
average_bedrooms = housing_data["total_bedrooms"].mean()
average_bedrooms
# fill NaN with mean value
housing_data["total_bedrooms"] = housing_data["total_bedrooms"].fillna(average_bedrooms)
# check if NaN still exist
housing_data["total_bedrooms"].isnull().value_counts()
# check NaN have been replaced by mean value
housing_data[housing_data["total_bedrooms"] == average_bedrooms]["total_bedrooms"]
# view dataframe to ensure
housing_data
# get categories in column
housing_data["ocean_proximity"].value_counts()
# encode categories as <1h ocean = 0, inland = 1, near ocean = 2, near bay = 3, island = 4
housing_data["ocean_proximity"] = housing_data["ocean_proximity"].map({"<1H OCEAN":0, "INLAND":1, "NEAR OCEAN":2, "NEAR BAY":3, "ISLAND":4})
# view cleaned dataset for model building
housing_data
# set x variable
x = housing_data.iloc[:, 0:-1]
x
# set y variable
y = pd.DataFrame(housing_data.iloc[:, -1])
y
# split x and y as 80% training and 20% test
x_train, x_test, y_train, y_test =  train_test_split(x,y,test_size=.20)
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)
# get scaler for standardizing x variates
scaler = StandardScaler()
# standardize only x variates using training mean and median to avoid corruption by test data
scaler.fit(x_train)
x_train_std = scaler.transform(x_train)
x_test_std = scaler.transform(x_test)
print(x_train_std.shape)
print(x_test_std.shape)
# fit linear regression model using standardized x_train
housing_price_predictor = LinearRegression()
housing_price_predictor.fit(x_train_std, y_train)
# predict using standardized x_test
y_predict = housing_price_predictor.predict(x_test_std)
# create table of predicted values
y_predict = pd.DataFrame(y_predict).rename(columns={0:"predicted house value"})
y_predict
# create table of test labels
y_test = y_test.reset_index().drop(columns="index")
# view y_test
y_test
# create combined table of y_predicted and y_test
y_predict = pd.concat([y_predict, y_test], axis=1)
y_predict
# round the predicted values
y_predict["predicted house value"] = y_predict["predicted house value"].apply(round)
# view final prediction table
y_predict
# calculate and show rmse value
rmse = metrics.mean_squared_error(y_predict["median_house_value"], y_predict["predicted house value"], squared=False)
rmse
# run regression only with median income as x-variate
x_train = x_train["median_income"]
x_test = x_test["median_income"]
# create tables for easy viewing
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)
# fit regression model
housing_price_predictor.fit(x_train, y_train)
# predict house prices using new model
y_predict = housing_price_predictor.predict(x_test)
# set predicted values in a table
y_predict = pd.DataFrame(y_predict).rename(columns={0:"predicted house value"})
y_predict
# view y labels for test data
y_test
# make summary table of prediction with actual values
y_predict = pd.concat([y_predict, y_test], axis=1)
y_predict["predicted house value"] = y_predict["predicted house value"].apply(round)
y_predict
# calculate and show rmse value
rmse = metrics.mean_squared_error(y_predict["median_house_value"], y_predict["predicted house value"], squared=False)
rmse
# plot training dataset with regression line
plt.scatter(x_train, y_train, color = "red", s=0.1)
plt.plot(x_train, housing_price_predictor.predict(x_train), color = "green")
plt.title("Median Income vs House Price (Training set)")
plt.xlabel("Median Income")
plt.ylabel("House Price Predicted")
plt.show()
# plot test dataset with regression line
plt.scatter(x_test, y_test, color = "red", s=0.1)
plt.plot(x_test, y_predict["predicted house value"], color = "green")
plt.title("Median Income vs House Price (Testing set)")
plt.xlabel("Median Income")
plt.ylabel("House Price Predicted")
plt.show()