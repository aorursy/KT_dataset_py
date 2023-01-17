import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

import warnings as ws

ws.defaultaction = "ignore"
print("lets Start")
df = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv")
df.head()
df.info()
df.shape
df.isna().sum()
df.model.value_counts().shape
df.transmission.value_counts().reset_index()
# Checking the price of car by transimission typ

price_by_transmission = df.groupby("transmission")['price'].mean().reset_index()

plt.title("Average Price of vechicle")

sns.set()

sns.barplot(x="transmission", y ="price", data = price_by_transmission)

plt.show()
milage_by_fuel = df.groupby("fuelType")["mileage"].mean().reset_index()

milage_by_fuel
cleaned_df = pd.concat([df,pd.get_dummies(df.fuelType), pd.get_dummies(df.transmission)], axis =1)
cleaned_df = cleaned_df.drop(["transmission", "fuelType","model"], axis = 1)
cleaned_df.head()
plt.figure(figsize = (15,15))

sns.heatmap(cleaned_df.corr(), annot = True)
abs (cleaned_df.corr()["price"]).sort_values(ascending =False)
# Fitting Regression Model

X = cleaned_df.drop("price", axis = 1)

y = cleaned_df["price"]
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state =1)
print(X_train.shape)

print(X_test.shape)

print (y_train.shape)

print (y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

y_pred =lin_reg.predict(X_test)

# Calculating RMSE

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

r2score = r2_score(y_test,y_pred)
print("R2 score is ", r2score)

print("rmse is ", rmse)
y_pred.shape

y_test.shape
temp = [i for i in  range(1, 1602)]

plt.style.use("seaborn")

plt.plot(temp,y_test,linestyle = "--", color = "green")

plt.plot(temp,y_pred,linestyle = "--", color = "red")

plt.xlabel("y_test VS y_pred")

plt.ylabel("temp value")

plt.legend()

plt.show()
#  Trying the Random Forest Regressior Model

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train,y_train)
forest_y_pred = forest.predict(X_test)

# Calculating RMSE

forest_rmse = np.sqrt(mean_squared_error(y_test,forest_y_pred))

forest_r2score = r2_score(y_test,forest_y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )
# Noe to model looks good 

temp = [i for i in  range(1, 1602)]

plt.figure(figsize = (26,10))

plt.style.use("seaborn")

plt.plot(temp,y_test,"--g")

plt.plot(temp,forest_y_pred,"--r")

plt.xlabel("y_test VS y_pred")

plt.ylabel("temp value")

plt.legend()

plt.show()
# Plotting proper chart using seaborn

resulting = pd.DataFrame(y_test)
resulting.reset_index().drop("index",axis =1)

resulting['prediction'] = forest_y_pred
resulting.head()
resulting.prediction = resulting.prediction.astype(int)
resulting.reset_index()