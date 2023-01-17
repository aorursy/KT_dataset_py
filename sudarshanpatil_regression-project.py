import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression 

from sklearn.preprocessing import StandardScaler

import warnings as ws

ws.defaultaction = "ignore"
df = pd.read_csv("/kaggle/input/housing.csv")
df.head()
df1 = df
df1.total_bedrooms = df.total_bedrooms.fillna(df.total_bedrooms.mean())
df1.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))

sns.heatmap(df1.corr(), annot= True)

plt.show()
np.abs( df1.corr()["median_house_value"].sort_values())
occean_prox_dummies = pd.get_dummies(df.ocean_proximity, drop_first = True)

occean_prox_dummies.head()
# Creating the baseline model 

X = df1[["total_rooms", "latitude", "median_income"]]

y = df["median_house_value"]
X.shape
y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size= 0.9, random_state = 1)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# Traning the Baseline Regression model 

lm = LinearRegression()

lm.fit(X_train, y_train)
print("The Coefficients are ", lm.coef_)

print("Intercept value if ", lm.intercept_)
y_pred = lm.predict(X_test)
y_pred.shape
temp = [i for i in range(1,2065)]
plt.figure(figsize = (30, 7))

plt.style.use("seaborn")

plt.plot (temp, y_test, color = "red", marker = "x")

plt.plot (temp, y_pred, color = "green", marker = "x")

plt.xlabel("y_test VS y_pred Comparision")

plt.show()
# Calculating the accuracy

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("The r2 score is ", r2_score(y_test,y_pred))
print("The mean sqaured error is ", np.sqrt(mean_squared_error(y_test,y_pred)))


df1.isnull().sum()
df1.head()
plt.style.use("seaborn")

df1.hist(bins= 20, figsize=(25,20))

plt.show()
# All the graphs as bottom heavy and they are on the different scale. Later we will try  to transform the data into the bell shaped curve
# As we an see the correlation coeficient of the median_income is quite high .i .e.  0.688075 and we should analyse it

df1.isnull().sum()
df1["income_cat"] = pd.cut(df1["median_income"],

                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],

                               labels=[1, 2, 3, 4, 5])

plt.hist(df1.income_cat)

plt.ylabel("Sallary Bins")

plt.xlabel("Count of the sallary")

plt.show()
# for finfing the relaion we are using the scatter matrix

from pandas.plotting import scatter_matrix



attributes = ["median_house_value", "median_income", "total_rooms",

              "housing_median_age"]

scatter_matrix(df1[attributes], figsize=(25, 15))

plt.show()
# Now  we are  deriving the new features from the old one
df1["rooms_per_household"] = df1["total_rooms"]/df1["households"]

df1["bedrooms_per_room"] = df1["total_bedrooms"]/df1["total_rooms"]

df1["population_per_household"]=df1["population"]/df1["households"]
df1.head()
plt.figure(figsize=(25,20))

sns.heatmap(df1.corr(), annot= True)

plt.show()
np.abs(df1.corr()["median_house_value"]).sort_values(ascending=False)
# Selecting the top 6 features 
df1.head()
df1.isnull().sum()
df1.shape