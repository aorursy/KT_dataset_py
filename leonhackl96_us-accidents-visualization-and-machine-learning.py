import numpy as np

import pandas as pd

import datetime

import gc



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
data = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")
data_size_mb = data.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % data_size_mb)
data.head(1)
data.shape
data.columns
data.isnull().sum().sort_values(ascending = False).head(25)
data = data.drop(["End_Lat", "End_Lng", "Precipitation(in)", "Number", 

                  "Wind_Chill(F)", "TMC", "Wind_Speed(mph)", "Nautical_Twilight", 

                  "Astronomical_Twilight", "Civil_Twilight", "Description", "ID", "Weather_Timestamp"], axis = 1)



data["Weather_Condition"].fillna(data["Weather_Condition"].mode()[0], inplace = True)

data["Visibility(mi)"].fillna(data["Visibility(mi)"].mode()[0], inplace = True)

data["Humidity(%)"].fillna(data["Humidity(%)"].mode()[0], inplace = True)

data["Temperature(F)"].fillna(data["Temperature(F)"].mode()[0], inplace = True)

data["Pressure(in)"].fillna(data["Pressure(in)"].mode()[0], inplace = True)

data["Wind_Direction"].fillna(data["Wind_Direction"].mode()[0], inplace = True)

data["Airport_Code"].fillna(data["Airport_Code"].mode()[0], inplace = True)

data["Timezone"].fillna(data["Timezone"].mode()[0], inplace = True)

data["Zipcode"].fillna(data["Zipcode"].mode()[0], inplace = True)

data["Sunrise_Sunset"].fillna(data["Sunrise_Sunset"].mode()[0], inplace = True)

data["City"].fillna(data["City"].mode()[0], inplace = True)



data.isnull().sum().sort_values(ascending = False).head(3)
data.shape
data.info()
plt.figure(figsize = (12,6))

sns.scatterplot(x = "Start_Lng", 

                y = "Start_Lat", 

                data = data, 

                hue = "State", 

                legend = False, 

                s = 15)



plt.title("USA")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Source").size().plot.bar()

plt.xticks(rotation = 360)

plt.title("Source of information")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Severity").size().plot.bar()

plt.xticks(rotation = 360)

plt.title("Severity of accidents")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("State").size().sort_values(ascending = False).plot.bar()

plt.xticks(rotation = 45)

plt.ylabel("Number of accidents")

plt.title("Number of accidents across states")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("City").size().sort_values(ascending = False).head(10).plot.bar()

plt.xticks(rotation = 25)

plt.ylabel("Number of accidents")

plt.title("Top 10 cities")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("County").size().sort_values(ascending = False).head(10).plot.bar()

plt.xticks(rotation = 25)

plt.ylabel("Number of accidents")

plt.title("Top 10 countys")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Weather_Condition").size().sort_values(ascending = False).head(10).plot.bar()

plt.xticks(rotation = 25)

plt.ylabel("Number of accidents")

plt.title("Most common weather conditions during accidents")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Visibility(mi)").size().sort_values(ascending = False).head(5).plot.bar()

plt.xticks(rotation = 360)

plt.ylabel("Number of accidents")

plt.title("Visibility during accidents")

plt.show()
data["Start_Time"] = pd.to_datetime(data["Start_Time"])

data["Year"] = data["Start_Time"].dt.year

data["Month"] = data["Start_Time"].dt.month

data["Hour"] = data["Start_Time"].dt.hour
plt.figure(figsize = (12,6))

data.groupby("Year").size().plot.bar()

plt.xticks(rotation = 360)

plt.ylabel("Number of accidents")

plt.title("Accidents per year (2016 - 2019)")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Month").size().plot.bar()

plt.xticks(rotation = 360)

plt.ylabel("Number of accidents")

plt.title("Accidents per month")

plt.show()
plt.figure(figsize = (12,6))

data.groupby("Hour").size().plot.bar()

plt.xticks(rotation = 360)

plt.ylabel("Number of accidents")

plt.title("Accidents per hour")

plt.show()
corr = data.corr()



plt.figure(figsize = (10,8))

sns.heatmap(corr, cmap = "coolwarm", linewidth = 2, linecolor = "white")

plt.title("Correlation")

plt.show()
gc.collect()
data = data.drop(["Zipcode", "Start_Time", "End_Time", "Street"], axis = 1)



wa_data = data.loc[data.State == "WA"]

wa_data = wa_data.drop("State", axis = 1)



ny_size_mb = wa_data.memory_usage().sum() / 1024 / 1024

print("Data memory size: %.2f MB" % ny_size_mb)
wa_data.shape
wa_data = pd.get_dummies(wa_data)
wa_data.shape
X = wa_data.drop("Severity", axis = 1)

Y = wa_data["Severity"]
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_pca = scaler.fit_transform(X)



pca = PCA(n_components = 2)

X_pca_transformed = pca.fit_transform(X_pca)



plt.figure(figsize = (12,6))



for i in Y.unique():

    X_pca_filtered = X_pca_transformed[Y == i, :]

    plt.scatter(X_pca_filtered[:, 0], X_pca_filtered[:, 1], s = 10, label = i, alpha = 0.5)

    

plt.legend()

plt.title("PCA")

plt.show()
gc.collect()
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
X_train, X_test, Y_train, Y_test = train_test_split(X, 

                                                    Y, 

                                                    random_state = 0, 

                                                    test_size = 0.25)
clf_xgb = XGBClassifier()

clf_xgb.fit(X_train, Y_train)



print(round(clf_xgb.score(X_test, Y_test), 4))



Y_predicted_xgb = clf_xgb.predict_proba(X_test)[:, 1]
clf_knn = KNeighborsClassifier()

clf_knn.fit(X_train, Y_train)



print(round(clf_knn.score(X_test, Y_test), 4))



Y_predicted_knn = clf_knn.predict_proba(X_test)[:, 1]
clf_rf = RandomForestClassifier()

clf_rf.fit(X_train, Y_train)



print(round(clf_rf.score(X_test, Y_test), 4))



Y_predicted_rf = clf_rf.predict_proba(X_test)[:, 1]