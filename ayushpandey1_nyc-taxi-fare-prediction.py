import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/nyctaxifares/NYCTaxiFares.csv')
data.head()
data.tail()
data.info()
# Description of all features
data.describe().T
data.isnull().sum()
data["fare_class"].value_counts()
print("Fare amount greater than 10$ :", data[data["fare_amount"]>=10].shape[0])
data[data["fare_amount"] >=10]
## Converting pickup_datetime from Object type to TimeStamp type

data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
data.head()
data.dtypes
from math import radians, cos,sin, asin,sqrt

def distance(lon1, lon2, lat1 , lat2):
    
    
    lon1 =radians(lon1)
    lon2 =radians(lon2)
    lat1 =radians(lat1)
    lat2 =radians(lat2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    r = 6371
    return(round (c * r, 2))
    
    
d = []
for i in range(data.shape[0]):
    d.append(distance(data["pickup_latitude"][i],
                      data["dropoff_latitude"][i],
                      data["pickup_longitude"][i],
                      data["dropoff_longitude"][i]))
data["distance in kms"] = d
data.head()
# Dropping Longitude and Latitude Features

data.drop(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"], axis=1, inplace=True)
data.head()
print("Date in data : ", data["pickup_datetime"].dt.day.sort_values().unique())
print("Month in data : ", data["pickup_datetime"].dt.month.unique()[0])
print("Year in data : ", data["pickup_datetime"].dt.year.unique()[0])
# Mapping days and Weekname
week_names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 5: "Friday", 6: "Saturday"}

data["weekday_name"] = data["pickup_datetime"].dt.weekday.map(week_names)
data.head()
plt.figure(figsize = (12,8))
data.groupby("weekday_name")["fare_amount"].sum().sort_values().plot()

plt.xlabel("Week", fontsize=15)
plt.ylabel("Fare Amount Average($)", fontsize=15)
plt.title("Total fare Amount vs Average", fontsize=20)
plt.show()
week_names_encode = {"Sunday": 1, "Saturday": 2, "Monday": 3, "Tuesday": 4, "Friday": 5, "Wednesday": 6, "Thursday": 7}
data["weekday_name"] = data["weekday_name"].map(week_names_encode)
data.head()
data["Hour"] = data["pickup_datetime"].dt.hour
data["Hour"].unique()

# Plotting graph of Fare vs Pickup time
plt.figure(figsize = (12,8))

data.groupby("Hour")["fare_amount"].sum().plot()
plt.title("Pickup Time vs Sum of Fare Amount at that Hour", fontsize=20)
plt.xlabel("Hour", fontsize=15)
plt.ylabel("Sum of Fare Amount", fontsize=15)
plt.show()
data["Month_Day"] = data["pickup_datetime"].dt.day
# Sum of Taxi Fare in a particular day

for day in list(data["pickup_datetime"].dt.day.sort_values().unique()):
    print(f"Date : {day} \t Total fare Amount : ${round(data[data.pickup_datetime.dt.day==day].fare_amount.sum(), 2)}")

plt.figure(figsize = (12, 8))

data.groupby("Month_Day")["fare_amount"].sum().plot()
plt.title("Pickup Time vs Sum of Fare Amount at that day", fontsize=20)
plt.xlabel("Month Day", fontsize=15)
plt.ylabel("Avg of Fare Amount", fontsize=15)
plt.show()
data.head()
data["passenger_count"].value_counts()
## Graph - Fare vs Distance

sns.relplot(data = data, kind = "scatter",x = "distance in kms",y = "fare_amount",
            hue = "passenger_count",height=6 ,aspect = 1.75,)
plt.title("Fare($) vs distance(kms)" , fontsize=15)
plt.show()

data.head()
data["fare_class"].value_counts()
data["fare_class"].unique()
# Total passenger travelling in a Taxi, paying Fare amount less than or more than $10.

data.groupby(["fare_class","passenger_count"])[["passenger_count"]].sum()
plt.figure(figsize=(15,8))
data.groupby("passenger_count")["fare_amount"].sum().sort_values().plot.barh()
plt.xlabel("Total Fare($)",fontsize =13)
plt.ylabel("Passengers in Taxi", fontsize =13)
plt.title("Number of passsenger vs Total Fare of Taxi", fontsize = 15)
plt.show()
data.drop("pickup_datetime", axis=1, inplace=True)
data.head()
data.to_csv("data_transformed.csv", index=False)
df = pd.read_csv("data_transformed.csv")
df.head()
# Separating dependent and independent feature
#### Dependent Feature ---> fare_amount

X = df.iloc[: , 1:]
y = df.iloc[: , 0]
X.head()
y.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.33)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression(fit_intercept= True, normalize =True)
linreg.fit(X_train , y_train)
y_pred = linreg.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test , y_pred)
from sklearn.ensemble import RandomForestRegressor
rfreg = RandomForestRegressor(n_estimators = 15)
rfreg.fit(X_train,y_train)
predict = rfreg.predict(X_test)
r2_score(y_test, predict)
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='mse', max_depth=None, random_state=42)
dt_reg.fit(X_train, y_train)
pred = dt_reg.predict(X_test)
r2_score(y_test, pred)
# Decision plot
from sklearn import tree
plt.figure(figsize = (15,8))
tree.plot_tree(dt_reg, max_depth = 2, fontsize = 15, feature_names=df.columns)
plt.title("<---------------------Decision Tree Split-------------------->", fontsize = 20)
plt.show()
from xgboost import XGBRegressor
xgb_reg = XGBRegressor(learning_rate= 0.30, max_depth=6, n_estimators=100, n_jobs =0)
xgb_reg.fit(X_train,y_train)
y_pred = xgb_reg.predict(X_test)
r2_score(y_pred, y_test)
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

n_estimators = [40,80,120,160]

criterion = ["mse","mae"]

max_depth = [int(x) for x in np.linspace(10,200,10)]

min_samples_split= [5,10,15]

min_samples_leaf = [4,6,8,10]

max_features = ['auto', 'sqrt', 'log2']
param_grid = {"n_estimators":n_estimators, "criterion":criterion, "max_depth":max_depth, "min_samples_split":
             min_samples_split, "min_samples_leaf":min_samples_leaf, "max_features":max_features}
rf_hyper = RandomForestRegressor()
rf_randomcv = RandomizedSearchCV(estimator=rf_hyper, param_distributions=param_grid, n_iter=10, 
                                 cv = 2, verbose=1, random_state=100, n_jobs=-1)
rf_randomcv.fit(X_train,y_train)
import pickle
filename = 'rf_NYCTaxifare_model.pkl'

pickle.dump(rfreg, open(filename,'wb'))
