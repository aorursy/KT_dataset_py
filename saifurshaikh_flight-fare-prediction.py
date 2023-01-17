# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_excel('../input/flight-fare-prediction-mh/Data_Train.xlsx', sheet_name=0)
test_df = pd.read_excel('../input/flight-fare-prediction-mh/Test_set.xlsx', sheet_name=0)
train_df.head()
train_df['Data'] = 'Train'
test_df['Data'] = 'Test'
test_df['Price'] = np.nan
test_df = test_df[train_df.columns]
df_all = pd.concat([train_df, test_df], axis=0)
df_all.shape
print(train_df.shape)
print(test_df.shape)
df_all.info()
df_all.head()
df_all.info()
df_all.Route.dropna(axis=0, inplace=True)
df_all.info()
# As we already have Total_stops column, dropping route column as it is of not much use
df_all.drop('Route', axis=1, inplace=True)
df_all.info()
df_all.Total_Stops.value_counts()
df_all.loc[df_all.Total_Stops.isnull()]
new_df = df_all.loc[(df_all.Source=='Delhi') & (df_all.Destination=='Cochin') & (df_all.Airline=='Air India') & (df_all.Price<10000)]
new_df
new_df.Total_Stops.value_counts()
# As we see that most of the Air India flights from Delhi to Cochin have 1 stop under price range of 10000, 
# I have replaced Nan value in Total_Stops to 1 Stop
df_all.Total_Stops.fillna('1 stop', inplace=True)
df_all.info()
df_all.head()
df_all['Date_of_Journey'] = pd.to_datetime(df_all['Date_of_Journey'])
df_all['Day_of_Journey'] = df_all.Date_of_Journey.dt.day
df_all['Month_of_Journey'] = df_all.Date_of_Journey.dt.month
df_all['Year_of_Journey'] = df_all.Date_of_Journey.dt.year
df_all.head()
df_all.drop('Date_of_Journey', axis=1, inplace=True)
df_all.info()
time = '23:30'
time
print(type(time))
time.split()[0]
df_all.Arrival_Time = df_all.Arrival_Time.str.split(expand=True)[0]
df_all.head()
df_all['Duration_hrs'] = df_all.Duration.str.split(expand=True)[0]
df_all['Duration_mins'] = df_all.Duration.str.split(expand=True)[1]
df_all.head()
# Time taken by plane to reach destination is called Duration
# It is the differnce betwwen Departure Time and Arrival time


# Assigning and converting Duration column into list
duration = list(df_all["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2:    # Check if duration contains only hour or mins
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour

duration_hrs = []
duration_mins = []
for i in range(len(duration)):
    duration_hrs.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
df_all.Duration_hrs = duration_hrs
df_all.Duration_mins = duration_mins
df_all.head()
df_all.drop('Duration', axis=1, inplace=True)
df_all['Dep_hr'] = pd.to_datetime(df_all.Dep_Time).dt.hour
df_all['Dep_min'] = pd.to_datetime(df_all.Dep_Time).dt.minute
df_all.drop('Dep_Time', axis=1, inplace=True)
df_all['Arrival_hr'] = pd.to_datetime(df_all.Arrival_Time).dt.hour
df_all['Arrival_min'] = pd.to_datetime(df_all.Arrival_Time).dt.minute
df_all.drop('Arrival_Time', axis=1, inplace=True)
df_all.head()
df_all.info()
df_all.select_dtypes('object').columns
df_all_dummy = pd.get_dummies(data=df_all, columns=['Airline', 'Source', 'Destination', 'Total_Stops', 'Additional_Info'], drop_first=True)
df_all_dummy.head()
train_set = df_all_dummy[df_all_dummy.Data=='Train']
test_set = df_all_dummy[df_all_dummy.Data=='Test']
train_set.drop('Data', axis=1, inplace=True)
test_set.drop(['Data','Price'], axis=1, inplace=True)
print(train_set.shape)
print(test_set.shape)
X = train_set.drop('Price', axis=1)
y = train_set['Price']
from sklearn.ensemble import ExtraTreesRegressor
etreg = ExtraTreesRegressor(n_estimators=500, max_features=10, random_state=2)
etreg.fit(X, y)
features_imp = pd.DataFrame(etreg.feature_importances_, index=X.columns, columns=['Score'])
features_imp.nlargest(20, columns=['Score']).index
X_imp_feat = X[['Total_Stops_non-stop', 'Airline_Jet Airways', 'Day_of_Journey',
       'Month_of_Journey', 'Duration_hrs', 'Airline_IndiGo',
       'Airline_Jet Airways Business',
       'Additional_Info_In-flight meal not included', 'Total_Stops_2 stops',
       'Additional_Info_No info', 'Destination_New Delhi', 'Arrival_hr',
       'Dep_hr', 'Dep_min', 'Duration_mins', 'Arrival_min', 'Airline_SpiceJet',
       'Destination_Delhi', 'Additional_Info_Business class',
       'Airline_Multiple carriers']]
X_imp_feat
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_imp_feat, y, test_size = 0.2, random_state=2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_hat = regressor.predict(X_test)
y_hat
from sklearn.metrics import mean_squared_error
import math
rmse = math.sqrt(mean_squared_error(y_test, y_hat))
rmse
param = {'alpha' : [0.00001, 0.0001, 0.001, 0.01, 0.1, 0, 1, 10, 100, 1000]}
from sklearn.model_selection import GridSearchCV
regressor1 = Lasso()
cross_val = GridSearchCV(regressor1, param_grid=param, cv=10, verbose=3)
cross_val.fit(X_train, y_train)
cross_val.best_estimator_
regressor1 = Lasso(alpha=0)
regressor1.fit(X_train, y_train)
y_hat1 = regressor1.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_hat1))
rmse
regressor2 = Ridge()
cross_val1 = GridSearchCV(regressor1, param_grid=param, cv=10, verbose=3)
cross_val1.fit(X_train, y_train)
cross_val1.best_params_
regressor1 = Ridge(alpha=0)
regressor2.fit(X_train, y_train)
y_hat2 = regressor2.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_hat2))
rmse
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
params = {'n_estimators' : [100, 300, 500, 700, 900, 1100, 1300, 1500],
         'max_features' : [5, 7, 10], 
         'max_depth' : [3, 4, 5, 6, 7], 
         'min_samples_split' : [10, 15, 20], 
         'min_samples_leaf' : [10, 15, 20]}
rf_reg = RandomForestRegressor(random_state=2)
random_cv = RandomizedSearchCV(rf_reg, param_distributions=params, cv=10, verbose=1)
random_cv.fit(X_train, y_train)
random_cv.best_estimator_
rf_reg = RandomForestRegressor(max_depth=7, max_features=10, min_samples_leaf=10,
                      min_samples_split=15, n_estimators=300, random_state=2)
rf_reg.fit(X_train, y_train)
y_hat = rf_reg.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_hat))
rmse
from sklearn import metrics
metrics.r2_score(y_test, y_hat)
rf_reg = RandomForestRegressor(random_state=2)
rf_reg.fit(X_train, y_train)
y_hat = rf_reg.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_hat))
print('RMSE: ',rmse)
r2 = metrics.r2_score(y_test, y_hat)
print('R-Square: ', r2)
