import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline
confirmed_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
deaths_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
recoveries_df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
#data1=pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
#confirmed_df.head()
#data1.head()
#data1.describe()

cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]

dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 
 
bd_cases = [] 

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recovered_sum = recoveries[i].sum()
    
    # confirmed, deaths, recovered, and active
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recovered_sum)
    total_active.append(confirmed_sum-death_sum-recovered_sum)
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recovered_sum/confirmed_sum)

    # case studies 
    bd_cases.append(confirmed_df[confirmed_df['Country/Region']=='Bangladesh'][i].sum())
    
def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

world_daily_increase = daily_increase(world_cases)

bd_daily_increase = daily_increase(bd_cases)

days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
print(bd_daily_increase)
print(bd_cases)
days_in_future = 30
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-30]

start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_1_22, bd_cases, test_size=0.2, shuffle=False)
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1)
svm_confirmed.fit(X_train_confirmed, y_train_confirmed)
svm_pred = svm_confirmed.predict(future_forcast)
# check against testing data
svm_test_pred = svm_confirmed.predict(X_test_confirmed)
plt.plot(y_test_confirmed)
plt.plot(svm_test_pred)
plt.legend(['Test Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svm_test_pred, y_test_confirmed))
print('MSE:',mean_squared_error(svm_test_pred, y_test_confirmed))
# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)
# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=False)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression predictions'])
# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(poly_X_train_confirmed, y_train_confirmed)

bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Regression Predictions'])

adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, bd_cases)
plt.title('Num of Coronavirus Cases Over Time (Total)', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('Num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, np.log10(bd_cases))
plt.title('Log of num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.bar(adjusted_dates, bd_daily_increase)
plt.title('BD Daily Increases in Confirmed Cases', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, bd_cases)
plt.plot(future_forcast, svm_pred, linestyle='dashed', color='red')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, bd_cases)
plt.plot(future_forcast, linear_pred, linestyle='dashed', color='black')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Days Since 1/22/2020', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
plt.figure(figsize=(5, 5))
plt.plot(adjusted_dates, bd_cases)
plt.plot(future_forcast, bayesian_pred, linestyle='dashed', color='green')
plt.title('num of Coronavirus Cases Over Time', size=12)
plt.xlabel('Time', size=12)
plt.ylabel('num of Cases', size=12)
plt.legend(['Confirmed Cases', 'Polynomial Bayesian Ridge Regression Predictions'], prop={'size': 10})
plt.xticks(size=10)
plt.yticks(size=10)
plt.show()
# Future predictions using SVM 
print('SVM future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(svm_pred[-30:])))
# Future predictions using Polynomial Regression 
linear_pred = linear_pred.reshape(1,-1)[0]
print('Polynomial regression future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(linear_pred[-30:])))
# Future predictions using Linear Regression 
print('Ridge regression future predictions for next 30 days:')
set(zip(future_forcast_dates[-30:], np.round(bayesian_pred[-30:])))