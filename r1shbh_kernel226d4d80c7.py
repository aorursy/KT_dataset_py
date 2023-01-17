import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel('../input/flight-price/Data_Train.xlsx')
data.head()
data.shape
data.isnull().any()
data.loc[data["Total_Stops"].isnull()]
#Checking what are the values that the Total_Stops take
data["Total_Stops"].value_counts()
#Filling the missing value with "non_stop"
data.loc[data["Total_Stops"].isnull(),"Total_Stops"] = "non-stop"
#Checking missing values again
data.isnull().any()
#Checking missing values in Route
data.loc[data["Route"].isnull()]
#Checking what are the values that the Route take
data["Route"].value_counts()
#Filling the missing value in Route variable
data.loc[data["Route"].isnull(),"Route"] = "DEL → COK"
data["Total_Stops"].value_counts()
stops_map = {"non-stop": 0,
             "1 stop": 1,
             "2 stops": 2,
             "3 stops": 3,
             "4 stops": 4}
data["Total_Stops"] = data["Total_Stops"].map(stops_map)
data["Total_Stops"].value_counts()
route_df = data["Route"].str.split("→")
route_df
route_df[100][1:-1]
route_df.head()
for index in data.index:
    data.loc[index, "Source"] = route_df[index][0].strip()
    data.loc[index, "Destination"] = route_df[index][-1].strip()
    del route_df[index][0]
    del route_df[index][-1]
data.head()
route_df.head()
route_df = route_df.apply(pd.Series)
route_df.rename({0:'Stop_1', 1:'Stop_2', 2:'Stop_3', 3:'Stop_4'}, axis = 'columns', inplace = True)
route_df.head()
data.drop(labels=['Route'], axis= 'columns', inplace=True)
new_data = pd.concat([data, route_df], axis =1)
new_data.head()
#Converting Departure Time and Duration into datetime format
new_data["Dep_Time"] = pd.to_datetime(new_data["Date_of_Journey"]+ ' ' + new_data["Dep_Time"])
new_data["Dep_Time"]
new_data.drop(labels=["Date_of_Journey", "Arrival_Time"], axis= 'columns', inplace = True)
new_data.head()
#TODO: Convert Duration to proper datetime format using regex and datetime functions

#TODO: Check contents of Additional_Info

#TODO: Seperate Price (y) and reemaing data (X)

#TODO: One Hot Encode these features: 'Airline', 'Additional_Info', 'Stop_<all>'

#Then the data will be ready to be used in any Regression model.
#Finding all numbers in a string
import re
duration_series = new_data["Duration"].str.findall(r'\d+')
duration_series
from datetime import datetime
from datetime import timedelta
for i, duration_list in enumerate(duration_series.values):
    hours = int(duration_list[0])
    if len(duration_list)==1:
        mins = 0
    else:
        mins = int(duration_list[1])
        
    new_data.loc[i, "Duration"] = 60*hours + mins
new_data.rename(columns={"Duration": "Minutes"}, inplace = True)
new_data.head()
new_data["Additional_Info"].value_counts()
new_data.loc[new_data["Additional_Info"]=="No Info", "Additional_Info"] = "No info"
new_data.head()
new_data.groupby('Airline')['Price'].mean().sort_values(ascending=False)
new_data['Airline'].value_counts()
for col in ['Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy','Trujet']:
    new_data = new_data.drop(new_data.index[new_data.Airline == col],  axis=0)
new_data.Airline.value_counts()
new_data.groupby('Airline')['Price'].mean().sort_values(ascending=False)
airline_dict = {}
for rank, key in enumerate(new_data.groupby('Airline')['Price'].mean().sort_values(ascending=False).keys()):
    airline_dict[key] = rank+1
airline_dict
new_data['Airline'] = new_data['Airline'].replace(airline_dict)
new_data.head()
new_data['Month'] = new_data['Dep_Time'].dt.month
new_data['Date'] = new_data['Dep_Time'].dt.day
new_data['Hour'] = new_data['Dep_Time'].dt.hour
new_data.head()
new_data.groupby('Source')['Price'].mean()
new_data.groupby('Destination')['Price'].mean()
new_data.describe()
new_data["Total_Stops"].hist()
new_data["Minutes"].plot()
new_data["Minutes"].hist()
#Less bins => More information lost
#So let us try to increase bins
new_data["Minutes"].hist(bins=30)
new_data["Minutes"].hist(bins=20)
new_data.groupby('Airline')['Price'].mean()
airline_grouped = new_data.groupby('Airline')['Price'].mean()

airline_grouped.plot.bar()
#Plotting again, without Jet Airways Business
new_data['Dep_Time'] = pd.to_datetime(new_data['Dep_Time'])
new_data.groupby(new_data["Dep_Time"].dt.month)["Price"].mean()
month_average = new_data.groupby(new_data["Dep_Time"].dt.month)["Price"].mean()
month_average.index = ["January", "March", "April", "May", "June", "September", "December"]
month_average.plot.bar()
new_data.groupby(new_data["Dep_Time"].dt.weekday_name)["Price"].mean()
days = ["Monday","Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_grouped = new_data.groupby(new_data["Dep_Time"].dt.weekday_name)["Price"].mean().reindex(days)
weekday_grouped.plot.bar()
new_data.groupby(new_data["Dep_Time"].dt.day)["Price"].mean()
day_grouped = new_data.groupby(new_data["Dep_Time"].dt.day)["Price"].mean()
day_grouped.plot.bar()
time_grouped = new_data.groupby(new_data["Dep_Time"].dt.hour)['Price'].mean()
time_grouped.plot.bar()
new_data.groupby('Additional_Info')['Price'].mean()
info_grouped = new_data.groupby('Additional_Info')['Price'].mean()
info_grouped.plot.bar()
new_data["Minutes"].values.max()
new_data["Minutes"].values.min()
NUM_BINS = (2860 - 75)//300
bins = np.linspace(75,2860,NUM_BINS) #Bins in the interval of 5 hours (300 mins)
hours_df = pd.cut(new_data['Minutes'], bins)
hours_df
new_data.groupby(hours_df)['Price'].mean()
new_data.groupby(hours_df)['Price'].mean().plot.bar()
new_data[["Minutes", "Price"]].corr()
source_grouped = new_data.groupby('Source')["Price"].mean()
source_grouped.plot.bar()
dest_grouped = new_data.groupby('Destination')["Price"].mean()
dest_grouped.plot.bar()
new_data.head()
#Creating a new dataframe
temp_df = new_data.copy()
temp_df["Dep_Day"] = new_data["Dep_Time"].dt.weekday.values
temp_df["Dep_Month"] = new_data["Dep_Time"].dt.month.values
temp_df["Dep_Hour"] = new_data["Dep_Time"].dt.hour.values
temp_df.drop(labels=["Dep_Time"], axis = 'columns', inplace = True)
temp_df.head()
temp_df.loc[temp_df["Additional_Info"] == "No info", "Additional_Info"] = np.nan
temp_df.head()
temp_df["Dep_Day"] = np.sin(2*np.pi*new_data["Dep_Time"].dt.day.values/7)
temp_df["Dep_Hour"] = np.sin(2*np.pi*new_data["Dep_Time"].dt.hour.values/24)
temp_df["Dep_Month"] = np.sin(2*np.pi*new_data["Dep_Time"].dt.month.values/12)
temp_df.head()
temp_df.rename({'Dep_Day':'Dep_Day_sine', 'Dep_Hour': 'Dep_Hour_sine', 'Dep_Month': 'Dep_Month_sine'},
               axis = 'columns', inplace = True)
temp_df.head()
from sklearn.preprocessing import OneHotEncoder
categorical_features = ['Airline', 'Source', 'Destination','Additional_Info', 'Stop_1', 
                                            'Stop_2', 'Stop_3', 'Stop_4']
temp_df.head()
one_hot = pd.get_dummies(temp_df[categorical_features], drop_first=True)
one_hot.head()
one_hot.shape
one_hot_df = pd.concat([temp_df, one_hot], axis = 1)
one_hot_df.drop(labels = categorical_features, axis = 'columns', inplace = True)
one_hot_df.head()
one_hot_df['Price'].hist()
#Taking log of prices
one_hot_df['Price'] = np.log(np.log(one_hot_df['Price']))
one_hot_df['Price'].hist()
#Rescaling Minutes
one_hot_df['Minutes'] /= one_hot_df['Minutes'].max()
one_hot_df.head()
y = one_hot_df["Price"].values
X = one_hot_df.drop(labels= ['Price'], axis = 'columns').values
X.shape, y.shape
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state = 2)
X_train.shape, y_train.shape
X_val.shape, y_val.shape
from sklearn.metrics import make_scorer
import math
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
def rmsle_score(y, y_pred):
    return 1 - rmsle(y, y_pred)
rmsle_scorer = make_scorer(rmsle_score, greater_is_better = True)
from sklearn.ensemble import RandomForestRegressor
rfg = RandomForestRegressor()
rfg.fit(X_train, y_train)
rfg.score(X_val, y_val)
rfg.score(X_train, y_train)
y_pred = rfg.predict(X_val)
1 - rmsle(y_val, y_pred)
1 - rmsle(rfg.predict(X_train), y_train)
from sklearn.linear_model import LinearRegression
lnr = LinearRegression()
lnr.fit(X_train, y_train)
lnr.score(X_train, y_train)
lnr.score(X_val, y_val)
y_pred = lnr.predict(X_val)
1 - rmsle(y_val, y_pred)
1 - rmsle(lnr.predict(X_train), y_train)
from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train, y_train)
1 - rmsle(rdg.predict(X_val), y_val)
1 - rmsle(rdg.predict(X_train), y_train)
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train, y_train)
1 - rmsle(lasso.predict(X_val), y_val)
1 - rmsle(lasso.predict(X_train), y_train)
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
1 - rmsle(svr.predict(X_val), y_val)
1 - rmsle(svr.predict(X_train), y_train)
RFnew = RandomForestRegressor(n_estimators=20)
RFnew.fit(X_train, y_train)
1 - rmsle(RFnew.predict(X_val), y_val)
1 - rmsle(RFnew.predict(X_train), y_train)
RFnew = RandomForestRegressor(n_estimators=100, max_depth = 20,min_samples_split = 15, max_features='sqrt',
                              min_samples_leaf = 1)
RFnew.fit(X_train, y_train)
1 - rmsle(RFnew.predict(X_val), y_val)
1 - rmsle(RFnew.predict(X_train), y_train)
param_rf = {'n_estimators':[80,110,140,170], 'max_depth':[17,18,19,20],
            'min_samples_split':[5,10,20,30,50,80], 'min_samples_leaf':[1,2,3],
            'max_features':['log2','sqrt','auto']}
from sklearn.model_selection import RandomizedSearchCV
gridRF = RandomizedSearchCV(estimator = RFnew, param_distributions = param_rf, scoring = rmsle_scorer, n_iter = 40)
gridRF.fit(X_train, y_train)
gridRF.best_params_
gridRF.score(X_train, y_train)
gridRF.score(X_val, y_val)
gridRF.best_score_
