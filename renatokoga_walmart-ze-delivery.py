# Load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import math
from sklearn.ensemble import RandomForestRegressor 
# Import files
TrainData = pd.read_csv("train.csv", parse_dates = ["Date"])
StoreData = pd.read_csv("stores.csv")
FeaturesData = pd.read_csv("features.csv", parse_dates = ["Date"])
TestData = pd.read_csv("test.csv", parse_dates = ["Date"])
StoreData.head()
# Extract day/month/year/weeknumber
TestData["Month"] = pd.DatetimeIndex(TestData['Date']).month
TestData["Year"] = pd.DatetimeIndex(TestData['Date']).year
TestData["Day"] = pd.DatetimeIndex(TestData['Date']).day
TestData["WeekNumber"] = TestData["Date"].dt.week
# Extract day/month/year/weeknumber
TrainData["Month"] = pd.DatetimeIndex(TrainData['Date']).month
TrainData["Year"] = pd.DatetimeIndex(TrainData['Date']).year
TrainData["Day"] = pd.DatetimeIndex(TrainData['Date']).day
TrainData["WeekNumber"] = TrainData["Date"].dt.week
TrainData.head()
# Extract day/month/year/weeknumber
FeaturesData["Month"] = pd.DatetimeIndex(FeaturesData['Date']).month
FeaturesData["Year"] = pd.DatetimeIndex(FeaturesData['Date']).year
FeaturesData["Day"] = pd.DatetimeIndex(FeaturesData['Date']).day
FeaturesData["WeekNumber"] = FeaturesData["Date"].dt.week
FeaturesData.head()
# Merge all data in a single dataframe
TrainData1 = pd.merge(TrainData, FeaturesData[["Store", "Date", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]], on=["Store", "Date"], how="left")
TrainData2 = pd.merge(TrainData1, StoreData, on=["Store"], how="left") 
TrainData2.head()
# Feature Engineering

# Holiday - convert to 1/0 format
TrainData2['IsHoliday'] = TrainData2['IsHoliday'].astype(int)

# Store type - as principal effect
TrainData2["Type_A"] = [1 if x == "A" else 0 for x in TrainData2["Type"]]
TrainData2["Type_B"] = [1 if x == "B" else 0 for x in TrainData2["Type"]]
TrainData2["Type_C"] = [1 if x == "C" else 0 for x in TrainData2["Type"]]

# Markdowns - replace missing values with 0
TrainData2[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]] = TrainData2[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].fillna(0)

# Markdown Total - sum of all markdowns
TrainData2["MarkDownTot"] = TrainData2["MarkDown1"] + TrainData2["MarkDown2"] + TrainData2["MarkDown3"] + TrainData2["MarkDown4"] + TrainData2["MarkDown5"]

# Black Friday and Christmas Day 
TrainData2["BF"] = [1 if x == 47 else 0 for x in TrainData2["WeekNumber"]]
TrainData2["Natal"] = [1 if x == 51 else 0 for x in TrainData2["WeekNumber"]]

TrainData2.head()
# Create datasets with dependent and independent variables for modeling
Y = TrainData2["Weekly_Sales"]
X = TrainData2[["Store", "Dept", "WeekNumber", "Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "MarkDownTot", "Type_A", "Type_B", "Type_C", "BF", "Natal"]]
# Split into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=19)
# Modeling using Random Forests
regressor = RandomForestRegressor(n_estimators=100, random_state=19)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)
# Checking model accuracy 1451
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
error = Y_test - Y_pred
# Validating model accuracy by independent variables
Y_pred_df = pd.DataFrame(data=Y_pred, columns=["Predicted"])
X_test_df = pd.merge(TrainData2, Y_pred_df, left_index = True, right_index = True, how="inner")
X_test_df["error"] = X_test_df["Weekly_Sales"] - X_test_df["Predicted"]
X_test_df.head()
# By Holiday
X_test_df.boxplot(column=["error"], by="IsHoliday", grid=False)
# Thanksgiving/Black Friday
X_test_df[X_test_df.BF == 1].boxplot(column=["error"], grid=False)
# Christmas
X_test_df[X_test_df.Natal == 1].boxplot(column=["error"], grid=False)
# By month
X_test_df.boxplot(column=["error"], by="Month", grid=False)
# By week
X_test_df[X_test_df.Month == 11].boxplot(column=["error"], by="WeekNumber", grid=False)
# Score evaluation dataset

TestData1 = pd.merge(TestData, FeaturesData[["Store", "Date", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "CPI", "Unemployment"]], on=["Store", "Date"], how="left")
TestData2 = pd.merge(TestData1, StoreData, on=["Store"], how="left") 


TestData2['IsHoliday'] = TestData2['IsHoliday'].astype(int)

TestData2["Type_A"] = [1 if x == "A" else 0 for x in TestData2["Type"]]
TestData2["Type_B"] = [1 if x == "B" else 0 for x in TestData2["Type"]]
TestData2["Type_C"] = [1 if x == "C" else 0 for x in TestData2["Type"]]

TestData2[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]] = TestData2[["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]].fillna(0)

TestData2["MarkDownTot"] = TestData2["MarkDown1"] + TestData2["MarkDown2"] + TestData2["MarkDown3"] + TestData2["MarkDown4"] + TestData2["MarkDown5"]

TestData2["BF"] = [1 if x == 47 else 0 for x in TestData2["WeekNumber"]]
TestData2["Natal"] = [1 if x == 51 else 0 for x in TestData2["WeekNumber"]]


Xk = TestData2[["Store", "Dept", "WeekNumber", "Size", "Temperature", "Fuel_Price", "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5", "MarkDownTot", "Type_A", "Type_B", "Type_C", "BF", "Natal"]]

Yk = regressor.predict(Xk)
Predicted = pd.DataFrame(data=Yk, columns=["Weekly_Sales"])
Predicted2 = pd.merge(TestData, Predicted, left_index = True, right_index = True)

# Create csv file to submit
Predicted2["Date2"] = Predicted2['Date'].astype(str)
Predicted2["Date3"] = Predicted2["Date2"].str[0:10]
Predicted2["Id"] =  Predicted2['Store'].astype(str) + '_' +  Predicted2['Dept'].astype(str) + '_' +  Predicted2['Date3'].astype(str)

Submit = Predicted2[["Id", "Weekly_Sales"]]
Submit.head()
Submit.to_csv('submit5.csv', index=False)
