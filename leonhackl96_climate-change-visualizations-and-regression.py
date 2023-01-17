import numpy as np

import pandas as pd

import datetime



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style

from matplotlib.pyplot import plot



style.use("seaborn-whitegrid")

%matplotlib inline

%config InlineBackend.figure_format = "retina"
df_global = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv")

df_country = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv")
data_size_mb_df_global = df_global.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size_mb_df_global)
data_size_mb_df_country = df_country.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size_mb_df_country)
df_global.head(3)
df_country.head(3)
print(df_global.shape)

print(df_country.shape)
df_global.info()
df_country.info()
df_global.isnull().sum()
df_country.isnull().sum()
df_global["dt"] = pd.to_datetime(df_global["dt"])

df_global["Month"] = df_global["dt"].dt.month

df_global["Year"] = df_global["dt"].dt.year

df_global = df_global.drop("dt", axis = 1)

df_global = df_global[df_global.Year >= 1900]



print(df_global.shape)
df_country["dt"] = pd.to_datetime(df_country["dt"])

df_country["Month"] = df_country["dt"].dt.month

df_country["Year"] = df_country["dt"].dt.year

df_country = df_country.drop("dt", axis = 1)

df_country = df_country[df_country.Year >= 1900]



print(df_country.shape)
df_global.isnull().sum().sort_values(ascending = False).head(3)
df_country.isnull().sum().sort_values(ascending = False).head(3)
df_country = df_country.dropna()

print(df_country.isnull().sum().sort_values(ascending = False).head(3))

print(df_country.shape)
plt.figure(figsize = (12,8))

sns.lineplot(data = df_global, x = "Year", y = "LandAndOceanAverageTemperature")

plt.title("Global Average Temperature per Year")

plt.ylabel("Temperature")

plt.show()
plt.figure(figsize = (12,8))

sns.lineplot(data = df_global, x = "Year", y = "LandMaxTemperature", label = "Max")

sns.lineplot(data = df_global, x = "Year", y = "LandMinTemperature", label = "Min")

plt.title("Global Maximum and Minimum Temperatures")

plt.ylabel("Temperature")

plt.legend()

plt.show()
plt.figure(figsize = (12,8))

df_country.groupby("Country")["AverageTemperature"].mean().sort_values(ascending = False).head(30).plot.bar()

plt.ylabel("Temperature")

plt.title("Hottest Countries")

plt.show()
plt.figure(figsize = (12,8))

df_country.groupby("Country")["AverageTemperature"].mean().sort_values(ascending = True).head(30).plot.bar()

plt.ylabel("Temperature")

plt.title("Coldest Countries")

plt.show()
plt.figure(figsize = (12,8))

sns.lineplot(data = df_country[df_country.Country == "Germany"], x = "Year", y = "AverageTemperature", ci = None, label = "Germany")

sns.lineplot(data = df_country[df_country.Country == "United States"], x = "Year", y = "AverageTemperature", ci = None, label = "United States")

sns.lineplot(data = df_country[df_country.Country == "Australia"], x = "Year", y = "AverageTemperature", ci = None, label = "Australia")

sns.lineplot(data = df_country[df_country.Country == "Norway"], x = "Year", y = "AverageTemperature", ci = None, label = "Norway")

sns.lineplot(data = df_country[df_country.Country == "South Africa"], x = "Year", y = "AverageTemperature", ci = None, label = "South Africa")

plt.ylabel("Temperature")

plt.legend()

plt.title("Average Temperature per Year")

plt.show()
df_ml = df_global.groupby("Year")["LandAndOceanAverageTemperature"].mean()

df_ml.head()
df_ml = df_ml.to_frame().reset_index()

print(df_ml.head(3))

print(df_ml.shape)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures
X = df_ml[["Year"]].values

Y = df_ml[["LandAndOceanAverageTemperature"]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, 

                                                    Y, 

                                                    random_state = 0, 

                                                    test_size = 0.25)
lm = LinearRegression()

lm.fit(X_train, Y_train)

print("Test accuracy: " + str(lm.score(X_test, Y_test)))

print("Train accuracy: " + str(lm.score(X_train, Y_train)))



Y_predicted_lm = lm.predict(X_test)
plt.figure(figsize = (12,8))

plt.scatter(X_train, Y_train, label = "Train")

plt.scatter(X_test, Y_test, label = "Test")

plt.plot(X_test, Y_predicted_lm, color = "green", linewidth = 2, label = "Regression")

plt.xlabel("Year")

plt.ylabel("Temperature")

plt.title("Linear Regression", fontsize = 16)

plt.legend()

plt.show()
poly = PolynomialFeatures(degree = 3, include_bias = False)



X_train_transformed = poly.fit_transform(X_train)

X_test_transformed = poly.fit_transform(X_test)
model_poly = LinearRegression()

model_poly.fit(X_train_transformed, Y_train)

print("Test accuracy: " + str(model_poly.score(X_test_transformed, Y_test)))

print("Train accuracy: " + str(model_poly.score(X_train_transformed, Y_train)))
plt.figure(figsize = (12,8))

plt.scatter(X_train, Y_train, label = "Train")

plt.scatter(X_test, Y_test, label = "Test")

plt.plot(X, model_poly.predict(poly.fit_transform(X)), color = "green", linewidth = 2, label = "Regression")

plt.xlabel("Year")

plt.ylabel("Temperature")

plt.title("Polynomial Regression", fontsize = 16)

plt.legend()

plt.show()