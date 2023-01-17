#Importing all the important Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as ncolors
import random
import math
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator

plt.style.use('seaborn')

%matplotlib inline
#Loading all the 3 dataset
confirmed = pd.read_csv("../input/covid-19-dataset/confirmed.csv")  
recovered = pd.read_csv("../input/covid-19-dataset/recovered.csv")  
death = pd.read_csv("../input/covid-19-dataset/death.csv")  
#Display the head of dataset
confirmed.head()
recovered.head()
death.head()
#Extracting all the columns using .key()
cols = confirmed.keys()
cols
confirmed1 = confirmed.loc[:, cols[4]:cols[-1]]
death1 = death.loc[:, cols[4]:cols[-1]]
recovered1 = recovered.loc[:, cols[4]:cols[-1]]
confirmed1.head()
death1.head()
recovered1.head()
dates = confirmed1.keys()
total_deaths = []
world_cases = []
active_cases = []
mortality_rate = []
total_recovered = []
for i in dates:
    confirmed_sum = confirmed1[i].sum()
    recovered_sum = recovered1[i].sum()
    world_cases.append(confirmed_sum)
    death_sum = death1[i].sum()
    total_deaths.append(death_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    total_recovered.append(recovered_sum)
confirmed_sum
dates
death_sum
recovered_sum
world_cases
#Convert all number in date format
day_since_1_18 = np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases = np.array(world_cases).reshape(-1,1)
total_deaths = np.array(total_deaths).reshape(-1,1)
total_recovered = np.array(total_recovered).reshape(-1,1)
day_since_1_18.shape
world_cases.shape
total_deaths.shape
total_recovered.shape
days_in_future = 10
future_forecast  = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast
len(future_forecast) #118 days
start = '01/04/2020'
start_date = datetime.datetime.strptime(start, '%d/%m/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%d/%m/%Y'))
future_forecast_dates
X_train, X_test, y_train, y_test = train_test_split(day_since_1_18,world_cases, test_size=0.15, shuffle=False)
#SVM
kernel = ['poly', 'sigmoid','rbf']
c = [0.01, 0.1, 1, 10]
gamma = [0.01, 0.1, 1]
epsilon = [0.01, 0.1, 1]
shrinking = [True, False]
svm_grid = {'kernel': kernel, 'C': c, 'gamma': gamma, 'epsilon': epsilon, 'shrinking': shrinking}

svm = SVR()
svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=20, verbose=1)
svm_search.fit(X_train, y_train)
svm_confirmed = svm_search.best_estimator_
svm_pred = svm_confirmed.predict(future_forecast)
svm_confirmed
svm_pred
svm_test_pred = svm_confirmed.predict(X_test)
plt.plot(svm_test_pred)
plt.plot(y_test)
print('MAE:', mean_absolute_error(svm_test_pred, y_test))
print('MSE:', mean_squared_error(svm_test_pred, y_test))
#SVM
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days since 4/1/2020', size=30)
plt.ylabel('Number of Cases', size = 30)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
#Confirmed VS Prediction
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forecast, svm_pred, color='purple')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days since 4/1/2020', size=30)
plt.ylabel('Number of Cases', size = 30)
plt.legend(['Confirmed cases','SVM Predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
#prediction
print('svm_pred')
set(zip(future_forecast_dates[10:],svm_pred[10:]))
#Randomforest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
print({'bootstrap': True,
 'criterion': 'mse',
 'max_depth': None,
 'max_features': 'auto',
 'max_leaf_nodes': None,
 'min_impurity_decrease': 0.0,
 'min_impurity_split': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'min_weight_fraction_leaf': 0.0,
 'n_estimators': 10,
 'n_jobs': 1,
 'oob_score': False,
 'random_state': 42,
 'verbose': 0,
 'warm_start': False})
labels = y_train
features = X_train
from pprint import pprint
#rf - start wait. yeah cell dekhne do ek baar-ok... ok done aage badho
n_estimators= [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split =  [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features' : max_features, 'max_depth' : max_depth, 'min_samples_split' : min_samples_split,'min_samples_leaf' : min_samples_leaf, 'bootstrap' : bootstrap}
pprint(random_grid)
rf = RandomForestRegressor()
rf_search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_search.fit(features, labels)
print(rf_search.best_params_)
rf_confirmed = rf_search.best_estimator_
rf_pred = rf_confirmed.predict(future_forecast)
len(rf_pred)
rfc_test_pred = rf_confirmed.predict(X_test)
plt.plot(rfc_test_pred)
plt.plot(y_test)
print('MAE:', mean_absolute_error(rfc_test_pred, y_test))
print('MSE:', mean_squared_error(rfc_test_pred, y_test))
#Linear Regression
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize=True, fit_intercept=True)
linear_model.fit(X_train, y_train)
test_linear_pred = linear_model.predict(X_test)
linear_pred = linear_model.predict(future_forecast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test))
print('MSE:', mean_squared_error(test_linear_pred, y_test))
plt.plot(y_test)
plt.plot(test_linear_pred)
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases)
plt.plot(future_forecast, linear_pred,color='orange')
plt.title('Number of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days since 4/1/2020', size=30)
plt.ylabel('Number Cases', size = 30)
plt.legend(['Confirmed Cases', 'Linear Regression Prediction'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()
print('linear regression future prediction')
print(linear_pred[-10:])
