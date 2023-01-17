import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
train_data = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx')
train_data.head()
train_data.info()
train_data.dropna(inplace = True)
train_data.isnull().sum()
train_data.drop(['Arrival_Time'],axis=1, inplace=True)
# date of journey to date time obj and taking date adn month out

train_data["Journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format="%d/%m/%Y").dt.day

train_data["Journey_month"] = pd.to_datetime(train_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month

train_data.drop(["Date_of_Journey"], axis = 1, inplace = True)
train_data.head()
# Depature Time

train_data["Dep_hour"] = pd.to_datetime(train_data["Dep_Time"]).dt.hour



train_data["Dep_min"] = pd.to_datetime(train_data["Dep_Time"]).dt.minute
train_data.drop(["Dep_Time"], axis = 1, inplace = True)
train_data.head()

train_data.drop(['Duration'],axis=1, inplace =True)
train_data.head()
train_data["Airline"].value_counts()
# performing oneHotEncoding

Airline = train_data[["Airline"]]



Airline = pd.get_dummies(Airline, drop_first= True)



Airline.head()
train_data["Source"].value_counts()
#ONeHotEncoding

Source = train_data[["Source"]]



Source = pd.get_dummies(Source, drop_first= True)



Source.head()
train_data["Destination"].value_counts()
# OneHotEncoding

Destination = train_data[["Destination"]]



Destination = pd.get_dummies(Destination, drop_first = True)



Destination.head()
train_data["Route"]
# most of the "Additional_info" Coloumn is filled with no_info so droping that col and route, Total_stops are correlated to each other so removong one of them

train_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
train_data["Total_Stops"].value_counts()
# LAbelEncoder

train_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)
final_data = pd.concat([train_data, Airline, Source, Destination], axis = 1)
final_data.head()
final_data.drop(["Airline", "Source", "Destination"], axis = 1, inplace = True)
final_data.head()
final_data.shape
# features an labels

y=final_data.iloc[:,1]

y.head()
final_data.drop(["Price"],axis=1, inplace=True)

x=final_data
x.head(), x.shape
# Train Test Split



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
# comparing all regressor models using lazypredict

import lazypredict
from lazypredict.Supervised import LazyRegressor

reg = LazyRegressor(verbose=0,ignore_warnings=False, custom_metric=None )

models,predictions = reg.fit(X_train, X_test, y_train, y_test)
models
from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import HistGradientBoostingRegressor	

reg_rf = HistGradientBoostingRegressor()

reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
reg_rf.score(X_train, y_train)
reg_rf.score(X_test, y_test)
sns.distplot(y_test-y_pred)

plt.show()


plt.scatter(y_test, y_pred, alpha = 0.5)

plt.xlabel("y_test")

plt.ylabel("y_pred")

plt.show()
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

print('MSE:', metrics.mean_squared_error(y_test, y_pred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
# calculating RMSE



2090.5509/(max(y)-min(y))
metrics.r2_score(y_test, y_pred)
import pickle



file = open('flight_fare_new_model.pkl', 'wb')

pickle.dump(reg_rf, file)
model = open('./flight_fare_new_model.pkl','rb')

Hist = pickle.load(model)
y_prediction = Hist.predict(X_test)
metrics.r2_score(y_test, y_prediction)