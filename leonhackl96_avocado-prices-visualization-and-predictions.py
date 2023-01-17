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



style.use("fivethirtyeight")

%matplotlib inline

%config InlineBackend.figure_format = "retina"
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")
data_size_mb = df.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size_mb)
df.head(3)
df.shape
df.info()
df.isnull().sum().sort_values(ascending = False)
plt.figure(figsize = (12,6))

sns.distplot(df["AveragePrice"])

plt.title("Price distribution")

plt.show()
df["Date"] = pd.to_datetime(df["Date"])

df["Month"] = df["Date"].dt.month



plt.figure(figsize = (12,6))

sns.lineplot(data = df, x = "Month", y = "AveragePrice", hue = "type")

plt.title("Average price: conventional vs. organic (per month)")

plt.show()
plt.figure(figsize = (12,6))

sns.lineplot(data = df, x = "year", y = "AveragePrice", hue = "type")

plt.title("Average price: conventional vs. organic (per year)")

plt.show()
plt.figure(figsize = (12,6))

sns.barplot(data = df, x = "Month", y = "Total Volume", hue = "type")

plt.title("Total volume by month")

plt.show()
plt.figure(figsize = (12,6))

sns.barplot(data = df, x = "year", y = "Total Volume", hue = "type")

plt.title("Total volume by year")

plt.show()
plt.figure(figsize = (12,6))



sns.barplot(data = df[df.region != "TotalUS"], 

            x = "region", 

            y = "Total Volume", 

            hue = "type", 

            ci = None)



plt.title("Total volume by region")

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (12,6))

sns.barplot(data = df, x = "Month", y = "Small Bags", color = "blue", label = "Small Bags")

sns.barplot(data = df, x = "Month", y = "Large Bags", color = "orange", label = "Large Bags")

sns.barplot(data = df, x = "Month", y = "XLarge Bags", color = "green", label = "XLarge Bags")

plt.title("Bag sales per month")

plt.ylabel("Bags")

plt.legend()

plt.show()
plt.figure(figsize = (12,6))

sns.barplot(data = df, x = "year", y = "Small Bags", color = "blue", label = "Small Bags")

sns.barplot(data = df, x = "year", y = "Large Bags", color = "orange", label = "Large Bags")

sns.barplot(data = df, x = "year", y = "XLarge Bags", color = "green", label = "XLarge Bags")

plt.title("Bag sales per year")

plt.ylabel("Bags")

plt.legend()

plt.show()
df = df.drop(["Unnamed: 0", "Date"], axis = 1)

df["Month"] = df["Month"].astype("int32")
df = pd.get_dummies(df)
df.shape
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBRegressor

from sklearn.ensemble import VotingRegressor
X = df.drop("AveragePrice", axis = 1)

Y = df["AveragePrice"]
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, 

                                                    Y, 

                                                    random_state = 0, 

                                                    test_size = 0.25)
clf_xgb = XGBRegressor(n_estimators = 700, max_depth = 5)

clf_xgb.fit(X_train, Y_train)

print(round(clf_xgb.score(X_test, Y_test), 4))
clf_rf = RandomForestRegressor(n_estimators = 300)

clf_rf.fit(X_train, Y_train)

print(round(clf_rf.score(X_test, Y_test), 4))
vote = VotingRegressor(estimators = [("XGBoost", clf_xgb), 

                                     ("RandomForest", clf_rf)])



vote.fit(X_train, Y_train)

print(round(vote.score(X_test, Y_test), 4))