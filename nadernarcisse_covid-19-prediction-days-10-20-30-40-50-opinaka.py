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

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/07-18-2020.csv')
us_medical_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/07-18-2020.csv')
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

china_cases = [] 
italy_cases = []
us_cases = [] 
spain_cases = [] 
france_cases = [] 
germany_cases = [] 
uk_cases = [] 
russia_cases = [] 
brazil_cases = []
india_cases = []
peru_cases = [] 

china_deaths = [] 
italy_deaths = []
us_deaths = [] 
spain_deaths = [] 
france_deaths = [] 
germany_deaths = [] 
uk_deaths = [] 
russia_deaths = []
brazil_deaths = [] 
india_deaths = []
peru_deaths = []

china_recoveries = [] 
italy_recoveries = []
us_recoveries = [] 
spain_recoveries = [] 
france_recoveries = [] 
germany_recoveries = [] 
uk_recoveries = [] 
russia_recoveries = [] 
brazil_recoveries = [] 
india_recoveries = [] 
peru_recoveries = [] 

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
    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_df[confirmed_df['Country/Region']=='France'][i].sum())
    germany_cases.append(confirmed_df[confirmed_df['Country/Region']=='Germany'][i].sum())
    uk_cases.append(confirmed_df[confirmed_df['Country/Region']=='United Kingdom'][i].sum())
    russia_cases.append(confirmed_df[confirmed_df['Country/Region']=='Russia'][i].sum())
    brazil_cases.append(confirmed_df[confirmed_df['Country/Region']=='Brazil'][i].sum())
    india_cases.append(confirmed_df[confirmed_df['Country/Region']=='India'][i].sum())
    peru_cases.append(confirmed_df[confirmed_df['Country/Region']=='Peru'][i].sum())
    
    
    china_deaths.append(deaths_df[deaths_df['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_df[deaths_df['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_df[deaths_df['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_df[deaths_df['Country/Region']=='France'][i].sum())
    germany_deaths.append(deaths_df[deaths_df['Country/Region']=='Germany'][i].sum())
    uk_deaths.append(deaths_df[deaths_df['Country/Region']=='United Kingdom'][i].sum())
    russia_deaths.append(deaths_df[deaths_df['Country/Region']=='Russia'][i].sum())
    brazil_deaths.append(deaths_df[deaths_df['Country/Region']=='Brazil'][i].sum())
    india_deaths.append(deaths_df[deaths_df['Country/Region']=='India'][i].sum())
    peru_deaths.append(deaths_df[deaths_df['Country/Region']=='Peru'][i].sum())
    
    china_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='China'][i].sum())
    italy_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())
    spain_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Spain'][i].sum())
    france_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='France'][i].sum())
    germany_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Germany'][i].sum())
    uk_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='United Kingdom'][i].sum())
    russia_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Russia'][i].sum())
    brazil_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Brazil'][i].sum())
    india_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='India'][i].sum())
    peru_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Peru'][i].sum())
dates = confirmed.keys()
world_cases = []
total_deaths = [] 
mortality_rate = []
recovery_rate = [] 
total_recovered = [] 
total_active = [] 

china_cases = [] 
italy_cases = []
us_cases = [] 
spain_cases = [] 
france_cases = [] 
germany_cases = [] 
uk_cases = [] 
russia_cases = [] 
brazil_cases = []
india_cases = []
peru_cases = [] 

china_deaths = [] 
italy_deaths = []
us_deaths = [] 
spain_deaths = [] 
france_deaths = [] 
germany_deaths = [] 
uk_deaths = [] 
russia_deaths = []
brazil_deaths = [] 
india_deaths = []
peru_deaths = []

china_recoveries = [] 
italy_recoveries = []
us_recoveries = [] 
spain_recoveries = [] 
france_recoveries = [] 
germany_recoveries = [] 
uk_recoveries = [] 
russia_recoveries = [] 
brazil_recoveries = [] 
india_recoveries = [] 
peru_recoveries = [] 

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
    china_cases.append(confirmed_df[confirmed_df['Country/Region']=='China'][i].sum())
    italy_cases.append(confirmed_df[confirmed_df['Country/Region']=='Italy'][i].sum())
    us_cases.append(confirmed_df[confirmed_df['Country/Region']=='US'][i].sum())
    spain_cases.append(confirmed_df[confirmed_df['Country/Region']=='Spain'][i].sum())
    france_cases.append(confirmed_df[confirmed_df['Country/Region']=='France'][i].sum())
    germany_cases.append(confirmed_df[confirmed_df['Country/Region']=='Germany'][i].sum())
    uk_cases.append(confirmed_df[confirmed_df['Country/Region']=='United Kingdom'][i].sum())
    russia_cases.append(confirmed_df[confirmed_df['Country/Region']=='Russia'][i].sum())
    brazil_cases.append(confirmed_df[confirmed_df['Country/Region']=='Brazil'][i].sum())
    india_cases.append(confirmed_df[confirmed_df['Country/Region']=='India'][i].sum())
    peru_cases.append(confirmed_df[confirmed_df['Country/Region']=='Peru'][i].sum())
    
    
    china_deaths.append(deaths_df[deaths_df['Country/Region']=='China'][i].sum())
    italy_deaths.append(deaths_df[deaths_df['Country/Region']=='Italy'][i].sum())
    us_deaths.append(deaths_df[deaths_df['Country/Region']=='US'][i].sum())
    spain_deaths.append(deaths_df[deaths_df['Country/Region']=='Spain'][i].sum())
    france_deaths.append(deaths_df[deaths_df['Country/Region']=='France'][i].sum())
    germany_deaths.append(deaths_df[deaths_df['Country/Region']=='Germany'][i].sum())
    uk_deaths.append(deaths_df[deaths_df['Country/Region']=='United Kingdom'][i].sum())
    russia_deaths.append(deaths_df[deaths_df['Country/Region']=='Russia'][i].sum())
    brazil_deaths.append(deaths_df[deaths_df['Country/Region']=='Brazil'][i].sum())
    india_deaths.append(deaths_df[deaths_df['Country/Region']=='India'][i].sum())
    peru_deaths.append(deaths_df[deaths_df['Country/Region']=='Peru'][i].sum())
    
    china_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='China'][i].sum())
    italy_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Italy'][i].sum())
    us_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='US'][i].sum())
    spain_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Spain'][i].sum())
    france_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='France'][i].sum())
    germany_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Germany'][i].sum())
    uk_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='United Kingdom'][i].sum())
    russia_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Russia'][i].sum())
    brazil_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Brazil'][i].sum())
    india_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='India'][i].sum())
    peru_recoveries.append(recoveries_df[recoveries_df['Country/Region']=='Peru'][i].sum())
days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed00X, X_test_confirmed00X, y_train_confirmed00X, y_test_confirmed00X = train_test_split(days_since_1_22[10:], world_cases[10:], test_size=0.12, shuffle=False)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed00X = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed00X.fit(X_train_confirmed00X, y_train_confirmed00X)
svm_pred00X = svm_confirmed00X.predict(future_forcast)

# check against testing data
svm_test_pred00X = svm_confirmed00X.predict(X_test_confirmed00X)
print('MAE:', mean_absolute_error(svm_test_pred00X, y_test_confirmed00X))
print('MSE:',mean_squared_error(svm_test_pred00X, y_test_confirmed00X))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed00X = poly.fit_transform(X_train_confirmed00X)
poly_X_test_confirmed00X = poly.fit_transform(X_test_confirmed00X)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00X = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed00X = bayesian_poly00X.fit_transform(X_train_confirmed00X)
bayesian_poly_X_test_confirmed00X = bayesian_poly00X.fit_transform(X_test_confirmed00X)
bayesian_poly_future_forcast00X = bayesian_poly00X.fit_transform(future_forcast)

# polynomial regression
linear_model00X = LinearRegression(normalize=True, fit_intercept=False)
linear_model00X.fit(poly_X_train_confirmed00X, y_train_confirmed00X)
test_linear_pred00X = linear_model00X.predict(poly_X_test_confirmed00X)
linear_pred00X = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00X, y_test_confirmed00X))
print('MSE:',mean_squared_error(test_linear_pred00X, y_test_confirmed00X))

print(linear_model00X.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00X = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00X = BayesianRidge(fit_intercept=False)
bayesian_search00X = RandomizedSearchCV(bayesian00X, bayesian_grid00X, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00X.fit(bayesian_poly_X_train_confirmed00X, y_train_confirmed00X)

bayesian_search00X.best_params_

bayesian_confirmed = bayesian_search00X.best_estimator_
test_bayesian_pred00X = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed00X)
bayesian_pred00X = bayesian_confirmed.predict(bayesian_poly_future_forcast00X)
print('MAE:', mean_absolute_error(test_bayesian_pred00X, y_test_confirmed00X))
print('MSE:',mean_squared_error(test_bayesian_pred00X, y_test_confirmed00X))
#10 Confirmed
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00X, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00X, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00X, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_confirmed00XX, X_test_confirmed00XX, y_train_confirmed00XX, y_test_confirmed00XX = train_test_split(days_since_1_22[20:], world_cases[20:], test_size=0.12, shuffle=False)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed00XX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed00XX.fit(X_train_confirmed00XX, y_train_confirmed00XX)
svm_pred00XX = svm_confirmed00XX.predict(future_forcast)

# check against testing data
svm_test_pred00XX = svm_confirmed00XX.predict(X_test_confirmed00XX)
print('MAE:', mean_absolute_error(svm_test_pred00XX, y_test_confirmed00XX))
print('MSE:',mean_squared_error(svm_test_pred00XX, y_test_confirmed00XX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed00XX = poly.fit_transform(X_train_confirmed00XX)
poly_X_test_confirmed00XX = poly.fit_transform(X_test_confirmed00XX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed00XX = bayesian_poly00XX.fit_transform(X_train_confirmed00XX)
bayesian_poly_X_test_confirmed00XX = bayesian_poly00XX.fit_transform(X_test_confirmed00XX)
bayesian_poly_future_forcast00XX = bayesian_poly00XX.fit_transform(future_forcast)

# polynomial regression
linear_model00XX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XX.fit(poly_X_train_confirmed00XX, y_train_confirmed00XX)
test_linear_pred00XX = linear_model00X.predict(poly_X_test_confirmed00XX)
linear_pred00XX = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XX, y_test_confirmed00XX))
print('MSE:',mean_squared_error(test_linear_pred00XX, y_test_confirmed00XX))

print(linear_model00XX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XX = BayesianRidge(fit_intercept=False)
bayesian_search00XX = RandomizedSearchCV(bayesian00XX, bayesian_grid00XX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XX.fit(bayesian_poly_X_train_confirmed00XX, y_train_confirmed00XX)

bayesian_search00XX.best_params_

bayesian_confirmed = bayesian_search00XX.best_estimator_
test_bayesian_pred00XX = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed00XX)
bayesian_pred00XX = bayesian_confirmed.predict(bayesian_poly_future_forcast00XX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XX, y_test_confirmed00XX))
print('MSE:',mean_squared_error(test_bayesian_pred00XX, y_test_confirmed00XX))
#20 confirmed
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_confirmed00XXX, X_test_confirmed00XXX, y_train_confirmed00XXX, y_test_confirmed00XXX = train_test_split(days_since_1_22[30:], world_cases[30:], test_size=0.12, shuffle=False)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed00XXX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed00XXX.fit(X_train_confirmed00XXX, y_train_confirmed00XXX)
svm_pred00XXX = svm_confirmed00XXX.predict(future_forcast)

# check against testing data
svm_test_pred00XXX = svm_confirmed00XXX.predict(X_test_confirmed00XXX)
print('MAE:', mean_absolute_error(svm_test_pred00XXX, y_test_confirmed00XXX))
print('MSE:',mean_squared_error(svm_test_pred00XXX, y_test_confirmed00XXX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed00XXX = poly.fit_transform(X_train_confirmed00XXX)
poly_X_test_confirmed00XXX = poly.fit_transform(X_test_confirmed00XXX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XXX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed00XXX = bayesian_poly00XXX.fit_transform(X_train_confirmed00XXX)
bayesian_poly_X_test_confirmed00XXX = bayesian_poly00XXX.fit_transform(X_test_confirmed00XXX)
bayesian_poly_future_forcast00XXX = bayesian_poly00XXX.fit_transform(future_forcast)

# polynomial regression
linear_model00XXX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XXX.fit(poly_X_train_confirmed00XXX, y_train_confirmed00XXX)
test_linear_pred00XXX = linear_model00X.predict(poly_X_test_confirmed00XXX)
linear_pred00XXX = linear_model00XXX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XXX, y_test_confirmed00XXX))
print('MSE:',mean_squared_error(test_linear_pred00XXX, y_test_confirmed00XXX))

print(linear_model00XXX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XXX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XXX = BayesianRidge(fit_intercept=False)
bayesian_search00XXX = RandomizedSearchCV(bayesian00XXX, bayesian_grid00XXX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XXX.fit(bayesian_poly_X_train_confirmed00XXX, y_train_confirmed00XXX)

bayesian_search00XXX.best_params_

bayesian_confirmed = bayesian_search00XXX.best_estimator_
test_bayesian_pred00XXX = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed00XXX)
bayesian_pred00XXX = bayesian_confirmed.predict(bayesian_poly_future_forcast00XXX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XXX, y_test_confirmed00XXX))
print('MSE:',mean_squared_error(test_bayesian_pred00XXX, y_test_confirmed00XXX))
#30 confirmed
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XXX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XXX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XXX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_confirmed00LX, X_test_confirmed00LX, y_train_confirmed00LX, y_test_confirmed00LX = train_test_split(days_since_1_22[40:], world_cases[40:], test_size=0.12, shuffle=False)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed00LX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed00LX.fit(X_train_confirmed00LX, y_train_confirmed00LX)
svm_pred00LX = svm_confirmed00LX.predict(future_forcast)

# check against testing data
svm_test_pred00LX = svm_confirmed00LX.predict(X_test_confirmed00LX)
print('MAE:', mean_absolute_error(svm_test_pred00LX, y_test_confirmed00LX))
print('MSE:',mean_squared_error(svm_test_pred00LX, y_test_confirmed00LX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed00LX = poly.fit_transform(X_train_confirmed00LX)
poly_X_test_confirmed00LX = poly.fit_transform(X_test_confirmed00LX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00LX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed00LX = bayesian_poly00LX.fit_transform(X_train_confirmed00LX)
bayesian_poly_X_test_confirmed00LX = bayesian_poly00LX.fit_transform(X_test_confirmed00LX)
bayesian_poly_future_forcast00LX = bayesian_poly00LX.fit_transform(future_forcast)

# polynomial regression
linear_model00LX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00LX.fit(poly_X_train_confirmed00LX, y_train_confirmed00LX)
test_linear_pred00LX = linear_model00LX.predict(poly_X_test_confirmed00LX)
linear_pred00LX = linear_model00LX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00LX, y_test_confirmed00LX))
print('MSE:',mean_squared_error(test_linear_pred00LX, y_test_confirmed00LX))

print(linear_model00LX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00LX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00LX = BayesianRidge(fit_intercept=False)
bayesian_search00LX = RandomizedSearchCV(bayesian00LX, bayesian_grid00LX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00LX.fit(bayesian_poly_X_train_confirmed00LX, y_train_confirmed00LX)

bayesian_search00LX.best_params_

bayesian_confirmed = bayesian_search00LX.best_estimator_
test_bayesian_pred00LX = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed00LX)
bayesian_pred00LX = bayesian_confirmed.predict(bayesian_poly_future_forcast00LX)
print('MAE:', mean_absolute_error(test_bayesian_pred00LX, y_test_confirmed00LX))
print('MSE:',mean_squared_error(test_bayesian_pred00LX, y_test_confirmed00LX))
#40 confirmed
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00LX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00LX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00LX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_confirmed00L, X_test_confirmed00L, y_train_confirmed00L, y_test_confirmed00L = train_test_split(days_since_1_22[50:], world_cases[50:], test_size=0.12, shuffle=False)
# svm_confirmed = svm_search.best_estimator_
svm_confirmed00L = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_confirmed00L.fit(X_train_confirmed00L, y_train_confirmed00L)
svm_pred00L = svm_confirmed00L.predict(future_forcast)

# check against testing data
svm_test_pred00L = svm_confirmed00L.predict(X_test_confirmed00L)
print('MAE:', mean_absolute_error(svm_test_pred00L, y_test_confirmed00L))
print('MSE:',mean_squared_error(svm_test_pred00L, y_test_confirmed00L))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_confirmed00L = poly.fit_transform(X_train_confirmed00L)
poly_X_test_confirmed00L = poly.fit_transform(X_test_confirmed00L)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00L = PolynomialFeatures(degree=4)
bayesian_poly_X_train_confirmed00L = bayesian_poly00L.fit_transform(X_train_confirmed00L)
bayesian_poly_X_test_confirmed00L = bayesian_poly00L.fit_transform(X_test_confirmed00L)
bayesian_poly_future_forcast00L = bayesian_poly00L.fit_transform(future_forcast)

# polynomial regression
linear_model00L = LinearRegression(normalize=True, fit_intercept=False)
linear_model00L.fit(poly_X_train_confirmed00L, y_train_confirmed00L)
test_linear_pred00L = linear_model00L.predict(poly_X_test_confirmed00L)
linear_pred00L = linear_model00L.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00L, y_test_confirmed00L))
print('MSE:',mean_squared_error(test_linear_pred00L, y_test_confirmed00L))

print(linear_model00L.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00L = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00L = BayesianRidge(fit_intercept=False)
bayesian_search00L = RandomizedSearchCV(bayesian00L, bayesian_grid00L, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00L.fit(bayesian_poly_X_train_confirmed00L, y_train_confirmed00L)

bayesian_search00L.best_params_

bayesian_confirmed = bayesian_search00L.best_estimator_
test_bayesian_pred00L = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed00L)
bayesian_pred00L = bayesian_confirmed.predict(bayesian_poly_future_forcast00L)
print('MAE:', mean_absolute_error(test_bayesian_pred00L, y_test_confirmed00L))
print('MSE:',mean_squared_error(test_bayesian_pred00L, y_test_confirmed00L))
#50 confirmed
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, world_cases, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00L, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00L, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00L, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
# Future predictions using SVM 
svm_pred00X_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 10)': np.round(svm_pred00X[:-1])})
svm_pred00X_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00X_Confirmed_sub
linear_pred00X = linear_pred00X.reshape(1,-1)[0]
linear_pred00X_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 10)': np.round(linear_pred00X[:-1])})
linear_pred00X_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00X_Confirmed_sub
bayesian_pred00X_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Bayesian Ridge Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 10)': np.round(bayesian_pred00X[:-1])})
bayesian_pred00X_Confirmed_sub.to_csv('bayesian_pred00X_Confirmed_sub.csv', index = False)
bayesian_pred00X_Confirmed_sub
linear_pred00XX = linear_pred00X.reshape(1,-1)[0]
linear_pred00XX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 20)': np.round(linear_pred00XX[:-1])})
linear_pred00XX_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00XX_Confirmed_sub
# Future predictions using SVM 
svm_pred00X_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 10)': np.round(svm_pred00X[:-1])})
svm_pred00X_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00X_Confirmed_sub
linear_pred00XX = linear_pred00XX.reshape(1,-1)[0]
linear_pred00XX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 20)': np.round(linear_pred00XX[:-1])})
linear_pred00XX_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00XX_Confirmed_sub
bayesian_pred00XX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Bayesian Ridge Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 20)': np.round(bayesian_pred00XX[:-1])})
bayesian_pred00XX_Confirmed_sub.to_csv('bayesian_pred00X_Confirmed_sub.csv', index = False)
bayesian_pred00XX_Confirmed_sub
# Future predictions using SVM 
svm_pred00XXX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 30)': np.round(svm_pred00XXX[:-1])})
svm_pred00XXX_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00XXX_Confirmed_sub
linear_pred00XXX = linear_pred00XXX.reshape(1,-1)[0]
linear_pred00XXX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 30)': np.round(linear_pred00XXX[:-1])})
linear_pred00XXX_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00XXX_Confirmed_sub
bayesian_pred00XXX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Bayesian Ridge Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 30)': np.round(bayesian_pred00XXX[:-1])})
bayesian_pred00XXX_Confirmed_sub.to_csv('bayesian_pred00X_Confirmed_sub.csv', index = False)
bayesian_pred00XXX_Confirmed_sub
# Future predictions using SVM 
svm_pred00LX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 40)': np.round(svm_pred00LX[:-1])})
svm_pred00LX_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00LX_Confirmed_sub
linear_pred00LX = linear_pred00LX.reshape(1,-1)[0]
linear_pred00LX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 40)': np.round(linear_pred00LX[:-1])})
linear_pred00LX_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00LX_Confirmed_sub
bayesian_pred00LX_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Bayesian Ridge Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 40)': np.round(bayesian_pred00LX[:-1])})
bayesian_pred00LX_Confirmed_sub.to_csv('bayesian_pred00X_Confirmed_sub.csv', index = False)
bayesian_pred00LX_Confirmed_sub
# Future predictions using SVM 
svm_pred00L_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 50)': np.round(svm_pred00L[:-1])})
svm_pred00L_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00L_Confirmed_sub
linear_pred00L = linear_pred00L.reshape(1,-1)[0]
linear_pred00L_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Polynomial Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 50)': np.round(linear_pred00L[:-1])})
linear_pred00L_Confirmed_sub.to_csv('linear_pred00X_Confirmed_sub.csv', index = False)
linear_pred00L_Confirmed_sub
bayesian_pred00L_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'Bayesian Ridge Regression Predictions Predicted # of Confirmed Cases Worldwide (Input 50)': np.round(bayesian_pred00L[:-1])})
bayesian_pred00L_Confirmed_sub.to_csv('bayesian_pred00L_Confirmed_sub.csv', index = False)
bayesian_pred00L_Confirmed_sub
X_train_deaths00X, X_test_deaths00X, y_train_deaths00X, y_test_deaths00X = train_test_split(days_since_1_22[10:], total_deaths[10:], test_size=0.12, shuffle=False)
# svm_deaths= svm_search.best_estimator_
svm_deaths00X = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00X.fit(X_train_deaths00X, y_train_deaths00X)
svm_pred00X = svm_deaths00X.predict(future_forcast)

# check against testing data
svm_test_pred00X = svm_deaths00X.predict(X_test_deaths00X)
print('MAE:', mean_absolute_error(svm_test_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(svm_test_pred00X, y_test_deaths00X))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00X = poly.fit_transform(X_train_deaths00X)
poly_X_test_deaths00X = poly.fit_transform(X_test_deaths00X)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00X = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00X = bayesian_poly00X.fit_transform(X_train_deaths00X)
bayesian_poly_X_test_deaths00X = bayesian_poly00X.fit_transform(X_test_deaths00X)
bayesian_poly_future_forcast00X = bayesian_poly00X.fit_transform(future_forcast)

# polynomial regression
linear_model00X = LinearRegression(normalize=True, fit_intercept=False)
linear_model00X.fit(poly_X_train_deaths00X, y_train_deaths00X)
test_linear_pred00X = linear_model00X.predict(poly_X_test_deaths00X)
linear_pred00X = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(test_linear_pred00X, y_test_deaths00X))

print(linear_model00X.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00X = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00X = BayesianRidge(fit_intercept=False)
bayesian_search00X = RandomizedSearchCV(bayesian00X, bayesian_grid00X, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00X.fit(bayesian_poly_X_train_deaths00X, y_train_deaths00X)

bayesian_search00X.best_params_

bayesian_deaths = bayesian_search00X.best_estimator_
test_bayesian_pred00X = bayesian_deaths.predict(bayesian_poly_X_test_deaths00X)
bayesian_pred00X = bayesian_deaths.predict(bayesian_poly_future_forcast00X)
print('MAE:', mean_absolute_error(test_bayesian_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(test_bayesian_pred00X, y_test_deaths00X))
#10 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00X, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00X, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00X, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Death Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00XX, X_test_deaths00XX, y_train_deaths00XX, y_test_deaths00XX = train_test_split(days_since_1_22[20:], total_deaths[20:], test_size=0.12, shuffle=False)
# svm_deaths = svm_search.best_estimator_
svm_deaths00XX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00XX.fit(X_train_deaths00XX, y_train_deaths00XX)
svm_pred00XX = svm_deaths00XX.predict(future_forcast)

# check against testing data
svm_test_pred00XX = svm_deaths00XX.predict(X_test_deaths00XX)
print('MAE:', mean_absolute_error(svm_test_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(svm_test_pred00XX, y_test_deaths00XX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00XX = poly.fit_transform(X_train_deaths00XX)
poly_X_test_deaths00XX = poly.fit_transform(X_test_deaths00XX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00XX = bayesian_poly00XX.fit_transform(X_train_deaths00XX)
bayesian_poly_X_test_deaths00XX = bayesian_poly00XX.fit_transform(X_test_deaths00XX)
bayesian_poly_future_forcast00XX = bayesian_poly00XX.fit_transform(future_forcast)

# polynomial regression
linear_model00XX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XX.fit(poly_X_train_deaths00XX, y_train_deaths00XX)
test_linear_pred00XX = linear_model00X.predict(poly_X_test_deaths00XX)
linear_pred00XX = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(test_linear_pred00XX, y_test_deaths00XX))

print(linear_model00XX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XX = BayesianRidge(fit_intercept=False)
bayesian_search00XX = RandomizedSearchCV(bayesian00XX, bayesian_grid00XX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XX.fit(bayesian_poly_X_train_deaths00XX, y_train_deaths00XX)

bayesian_search00XX.best_params_

bayesian_deaths = bayesian_search00XX.best_estimator_
test_bayesian_pred00XX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00XX)
bayesian_pred00XX = bayesian_deaths.predict(bayesian_poly_future_forcast00XX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(test_bayesian_pred00XX, y_test_deaths00XX))
#20 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Death Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00XXX, X_test_deaths00XXX, y_train_deaths00XXX, y_test_deaths00XXX = train_test_split(days_since_1_22[30:], total_deaths[30:], test_size=0.12, shuffle=False)
# svm_deaths = svm_search.best_estimator_
svm_deaths00XXX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00XXX.fit(X_train_deaths00XXX, y_train_deaths00XXX)
svm_pred00XXX = svm_deaths00XXX.predict(future_forcast)

# check against testing data
svm_test_pred00XXX = svm_deaths00XXX.predict(X_test_deaths00XXX)
print('MAE:', mean_absolute_error(svm_test_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(svm_test_pred00XXX, y_test_deaths00XXX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00XXX = poly.fit_transform(X_train_deaths00XXX)
poly_X_test_deaths00XXX = poly.fit_transform(X_test_deaths00XXX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XXX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00XXX = bayesian_poly00XXX.fit_transform(X_train_deaths00XXX)
bayesian_poly_X_test_deaths00XXX = bayesian_poly00XXX.fit_transform(X_test_deaths00XXX)
bayesian_poly_future_forcast00XXX = bayesian_poly00XXX.fit_transform(future_forcast)

# polynomial regression
linear_model00XXX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XXX.fit(poly_X_train_deaths00XXX, y_train_deaths00XXX)
test_linear_pred00XXX = linear_model00X.predict(poly_X_test_deaths00XXX)
linear_pred00XXX = linear_model00XXX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(test_linear_pred00XXX, y_test_deaths00XXX))

print(linear_model00XXX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XXX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XXX = BayesianRidge(fit_intercept=False)
bayesian_search00XXX = RandomizedSearchCV(bayesian00XXX, bayesian_grid00XXX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XXX.fit(bayesian_poly_X_train_deaths00XXX, y_train_deaths00XXX)

bayesian_search00XXX.best_params_

bayesian_deaths = bayesian_search00XXX.best_estimator_
test_bayesian_pred00XXX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00XXX)
bayesian_pred00XXX = bayesian_deaths.predict(bayesian_poly_future_forcast00XXX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(test_bayesian_pred00XXX, y_test_deaths00XXX))
#30 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XXX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XXX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XXX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Death Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00LX, X_test_deaths00LX, y_train_deaths00LX, y_test_deaths00LX = train_test_split(days_since_1_22[40:], total_deaths[40:], test_size=0.12, shuffle=False)
# svm_deaths = svm_search.best_estimator_
svm_deaths00LX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00LX.fit(X_train_deaths00LX, y_train_deaths00LX)
svm_pred00LX = svm_deaths00LX.predict(future_forcast)

# check against testing data
svm_test_pred00LX = svm_deaths00LX.predict(X_test_deaths00LX)
print('MAE:', mean_absolute_error(svm_test_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(svm_test_pred00LX, y_test_deaths00LX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00LX = poly.fit_transform(X_train_deaths00LX)
poly_X_test_deaths00LX = poly.fit_transform(X_test_deaths00LX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00LX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00LX = bayesian_poly00LX.fit_transform(X_train_deaths00LX)
bayesian_poly_X_test_deaths00LX = bayesian_poly00LX.fit_transform(X_test_deaths00LX)
bayesian_poly_future_forcast00LX = bayesian_poly00LX.fit_transform(future_forcast)

# polynomial regression
linear_model00LX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00LX.fit(poly_X_train_deaths00LX, y_train_deaths00LX)
test_linear_pred00LX = linear_model00LX.predict(poly_X_test_deaths00LX)
linear_pred00LX = linear_model00LX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(test_linear_pred00LX, y_test_deaths00LX))

print(linear_model00LX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00LX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00LX = BayesianRidge(fit_intercept=False)
bayesian_search00LX = RandomizedSearchCV(bayesian00LX, bayesian_grid00LX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00LX.fit(bayesian_poly_X_train_deaths00LX, y_train_deaths00LX)

bayesian_search00LX.best_params_

bayesian_deaths = bayesian_search00LX.best_estimator_
test_bayesian_pred00LX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00LX)
bayesian_pred00LX = bayesian_deaths.predict(bayesian_poly_future_forcast00LX)
print('MAE:', mean_absolute_error(test_bayesian_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(test_bayesian_pred00LX, y_test_deaths00LX))
#40 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00LX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00LX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00LX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Death Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00L, X_test_deaths00L, y_train_deaths00L, y_test_deaths00L = train_test_split(days_since_1_22[50:], total_deaths[50:], test_size=0.12, shuffle=False)
# svm_deaths = svm_search.best_estimator_
svm_deaths00L = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00L.fit(X_train_deaths00L, y_train_deaths00L)
svm_pred00L = svm_deaths00L.predict(future_forcast)

# check against testing data
svm_test_pred00L = svm_deaths00L.predict(X_test_deaths00L)
print('MAE:', mean_absolute_error(svm_test_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(svm_test_pred00L, y_test_deaths00L))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00L = poly.fit_transform(X_train_deaths00L)
poly_X_test_deaths00L = poly.fit_transform(X_test_deaths00L)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00L = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00L = bayesian_poly00L.fit_transform(X_train_deaths00L)
bayesian_poly_X_test_deaths00L = bayesian_poly00L.fit_transform(X_test_deaths00L)
bayesian_poly_future_forcast00L = bayesian_poly00L.fit_transform(future_forcast)

# polynomial regression
linear_model00L = LinearRegression(normalize=True, fit_intercept=False)
linear_model00L.fit(poly_X_train_deaths00L, y_train_deaths00L)
test_linear_pred00L = linear_model00L.predict(poly_X_test_deaths00L)
linear_pred00L = linear_model00L.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(test_linear_pred00L, y_test_deaths00L))

print(linear_model00L.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00L = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00L = BayesianRidge(fit_intercept=False)
bayesian_search00L = RandomizedSearchCV(bayesian00L, bayesian_grid00L, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00L.fit(bayesian_poly_X_train_deaths00L, y_train_deaths00L)

bayesian_search00L.best_params_

bayesian_deaths = bayesian_search00L.best_estimator_
test_bayesian_pred00L = bayesian_deaths.predict(bayesian_poly_X_test_deaths00L)
bayesian_pred00L = bayesian_deaths.predict(bayesian_poly_future_forcast00L)
print('MAE:', mean_absolute_error(test_bayesian_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(test_bayesian_pred00L, y_test_deaths00L))
#50 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_deaths, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00L, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00L, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00L, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Death Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
# Future predictions using SVM 
svm_pred00X_Confirmed_sub = pd.DataFrame({'Date': future_forcast_dates[:-1], 'SVM Predicted # of Confirmed Cases Worldwide (Input 10)': np.round(svm_pred00X[:-1])})
svm_pred00X_Confirmed_sub.to_csv('svm_pred00X_Confirmed_sub.csv', index = False)
svm_pred00X_Confirmed_sub
X_train_deaths00X, X_test_deaths00X, y_train_deaths00X, y_test_deaths00X = train_test_split(days_since_1_22[10:], total_recovered[10:], test_size=0.12, shuffle=False)
# svm_deaths= svm_search.best_estimator_
svm_deaths00X = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00X.fit(X_train_deaths00X, y_train_deaths00X)
svm_pred00X = svm_deaths00X.predict(future_forcast)

# check against testing data
svm_test_pred00X = svm_deaths00X.predict(X_test_deaths00X)
print('MAE:', mean_absolute_error(svm_test_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(svm_test_pred00X, y_test_deaths00X))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00X = poly.fit_transform(X_train_deaths00X)
poly_X_test_deaths00X = poly.fit_transform(X_test_deaths00X)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00X = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00X = bayesian_poly00X.fit_transform(X_train_deaths00X)
bayesian_poly_X_test_deaths00X = bayesian_poly00X.fit_transform(X_test_deaths00X)
bayesian_poly_future_forcast00X = bayesian_poly00X.fit_transform(future_forcast)

# polynomial regression
linear_model00X = LinearRegression(normalize=True, fit_intercept=False)
linear_model00X.fit(poly_X_train_deaths00X, y_train_deaths00X)
test_linear_pred00X = linear_model00X.predict(poly_X_test_deaths00X)
linear_pred00X = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(test_linear_pred00X, y_test_deaths00X))

print(linear_model00X.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00X = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00X = BayesianRidge(fit_intercept=False)
bayesian_search00X = RandomizedSearchCV(bayesian00X, bayesian_grid00X, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00X.fit(bayesian_poly_X_train_deaths00X, y_train_deaths00X)

bayesian_search00X.best_params_

bayesian_deaths = bayesian_search00X.best_estimator_
test_bayesian_pred00X = bayesian_deaths.predict(bayesian_poly_X_test_deaths00X)
bayesian_pred00X = bayesian_deaths.predict(bayesian_poly_future_forcast00X)
print('MAE:', mean_absolute_error(test_bayesian_pred00X, y_test_deaths00X))
print('MSE:',mean_squared_error(test_bayesian_pred00X, y_test_deaths00X))
#10 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00X, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00X, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00X, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Recovered Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00XX, X_test_deaths00XX, y_train_deaths00XX, y_test_deaths00XX = train_test_split(days_since_1_22[20:], total_recovered[20:], test_size=0.12, shuffle=False)
# svm_deaths = svm_search.best_estimator
svm_deaths00XX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00XX.fit(X_train_deaths00XX, y_train_deaths00XX)
svm_pred00XX = svm_deaths00XX.predict(future_forcast)

# check against testing data
svm_test_pred00XX = svm_deaths00XX.predict(X_test_deaths00XX)
print('MAE:', mean_absolute_error(svm_test_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(svm_test_pred00XX, y_test_deaths00XX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00XX = poly.fit_transform(X_train_deaths00XX)
poly_X_test_deaths00XX = poly.fit_transform(X_test_deaths00XX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00XX = bayesian_poly00XX.fit_transform(X_train_deaths00XX)
bayesian_poly_X_test_deaths00XX = bayesian_poly00XX.fit_transform(X_test_deaths00XX)
bayesian_poly_future_forcast00XX = bayesian_poly00XX.fit_transform(future_forcast)

# polynomial regression
linear_model00XX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XX.fit(poly_X_train_deaths00XX, y_train_deaths00XX)
test_linear_pred00XX = linear_model00X.predict(poly_X_test_deaths00XX)
linear_pred00XX = linear_model00X.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(test_linear_pred00XX, y_test_deaths00XX))

print(linear_model00XX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XX = BayesianRidge(fit_intercept=False)
bayesian_search00XX = RandomizedSearchCV(bayesian00XX, bayesian_grid00XX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XX.fit(bayesian_poly_X_train_deaths00XX, y_train_deaths00XX)

bayesian_search00XX.best_params_

bayesian_deaths = bayesian_search00XX.best_estimator_
test_bayesian_pred00XX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00XX)
bayesian_pred00XX = bayesian_deaths.predict(bayesian_poly_future_forcast00XX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XX, y_test_deaths00XX))
print('MSE:',mean_squared_error(test_bayesian_pred00XX, y_test_deaths00XX))
#20 deaths
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Recovered Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00XXX, X_test_deaths00XXX, y_train_deaths00XXX, y_test_deaths00XXX = train_test_split(days_since_1_22[30:], total_recovered[30:], test_size=0.12, shuffle=False)
# svm_recovered = svm_search.best_estimator_
svm_deaths00XXX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00XXX.fit(X_train_deaths00XXX, y_train_deaths00XXX)
svm_pred00XXX = svm_deaths00XXX.predict(future_forcast)

# check against testing data
svm_test_pred00XXX = svm_deaths00XXX.predict(X_test_deaths00XXX)
print('MAE:', mean_absolute_error(svm_test_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(svm_test_pred00XXX, y_test_deaths00XXX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00XXX = poly.fit_transform(X_train_deaths00XXX)
poly_X_test_deaths00XXX = poly.fit_transform(X_test_deaths00XXX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00XXX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00XXX = bayesian_poly00XXX.fit_transform(X_train_deaths00XXX)
bayesian_poly_X_test_deaths00XXX = bayesian_poly00XXX.fit_transform(X_test_deaths00XXX)
bayesian_poly_future_forcast00XXX = bayesian_poly00XXX.fit_transform(future_forcast)

# polynomial regression
linear_model00XXX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00XXX.fit(poly_X_train_deaths00XXX, y_train_deaths00XXX)
test_linear_pred00XXX = linear_model00X.predict(poly_X_test_deaths00XXX)
linear_pred00XXX = linear_model00XXX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(test_linear_pred00XXX, y_test_deaths00XXX))

print(linear_model00XXX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00XXX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00XXX = BayesianRidge(fit_intercept=False)
bayesian_search00XXX = RandomizedSearchCV(bayesian00XXX, bayesian_grid00XXX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00XXX.fit(bayesian_poly_X_train_deaths00XXX, y_train_deaths00XXX)

bayesian_search00XXX.best_params_

bayesian_deaths = bayesian_search00XXX.best_estimator_
test_bayesian_pred00XXX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00XXX)
bayesian_pred00XXX = bayesian_deaths.predict(bayesian_poly_future_forcast00XXX)
print('MAE:', mean_absolute_error(test_bayesian_pred00XXX, y_test_deaths00XXX))
print('MSE:',mean_squared_error(test_bayesian_pred00XXX, y_test_deaths00XXX))
#30 recovered
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00XXX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00XXX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00XXX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Recovered Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00LX, X_test_deaths00LX, y_train_deaths00LX, y_test_deaths00LX = train_test_split(days_since_1_22[40:], total_recovered[40:], test_size=0.12, shuffle=False)
# svm_recovered = svm_search.best_estimator_
svm_deaths00LX = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00LX.fit(X_train_deaths00LX, y_train_deaths00LX)
svm_pred00LX = svm_deaths00LX.predict(future_forcast)

# check against testing data
svm_test_pred00LX = svm_deaths00LX.predict(X_test_deaths00LX)
print('MAE:', mean_absolute_error(svm_test_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(svm_test_pred00LX, y_test_deaths00LX))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00LX = poly.fit_transform(X_train_deaths00LX)
poly_X_test_deaths00LX = poly.fit_transform(X_test_deaths00LX)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00LX = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00LX = bayesian_poly00LX.fit_transform(X_train_deaths00LX)
bayesian_poly_X_test_deaths00LX = bayesian_poly00LX.fit_transform(X_test_deaths00LX)
bayesian_poly_future_forcast00LX = bayesian_poly00LX.fit_transform(future_forcast)

# polynomial regression
linear_model00LX = LinearRegression(normalize=True, fit_intercept=False)
linear_model00LX.fit(poly_X_train_deaths00LX, y_train_deaths00LX)
test_linear_pred00LX = linear_model00LX.predict(poly_X_test_deaths00LX)
linear_pred00LX = linear_model00LX.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(test_linear_pred00LX, y_test_deaths00LX))

print(linear_model00LX.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00LX = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00LX = BayesianRidge(fit_intercept=False)
bayesian_search00LX = RandomizedSearchCV(bayesian00LX, bayesian_grid00LX, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00LX.fit(bayesian_poly_X_train_deaths00LX, y_train_deaths00LX)

bayesian_search00LX.best_params_

bayesian_deaths = bayesian_search00LX.best_estimator_
test_bayesian_pred00LX = bayesian_deaths.predict(bayesian_poly_X_test_deaths00LX)
bayesian_pred00LX = bayesian_deaths.predict(bayesian_poly_future_forcast00LX)
print('MAE:', mean_absolute_error(test_bayesian_pred00LX, y_test_deaths00LX))
print('MSE:',mean_squared_error(test_bayesian_pred00LX, y_test_deaths00LX))
#40 Recovered
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00LX, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00LX, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00LX, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Recovered Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()
X_train_deaths00L, X_test_deaths00L, y_train_deaths00L, y_test_deaths00L = train_test_split(days_since_1_22[50:], total_recovered[50:], test_size=0.12, shuffle=False)
# svm_recovered = svm_search.best_estimator_
svm_deaths00L = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=3, C=0.1)
svm_deaths00L.fit(X_train_deaths00L, y_train_deaths00L)
svm_pred00L = svm_deaths00L.predict(future_forcast)

# check against testing data
svm_test_pred00L = svm_deaths00L.predict(X_test_deaths00L)
print('MAE:', mean_absolute_error(svm_test_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(svm_test_pred00L, y_test_deaths00L))

# transform our data for polynomial regression
poly = PolynomialFeatures(degree=5)
poly_X_train_deaths00L = poly.fit_transform(X_train_deaths00L)
poly_X_test_deaths00L = poly.fit_transform(X_test_deaths00L)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly00L = PolynomialFeatures(degree=4)
bayesian_poly_X_train_deaths00L = bayesian_poly00L.fit_transform(X_train_deaths00L)
bayesian_poly_X_test_deaths00L = bayesian_poly00L.fit_transform(X_test_deaths00L)
bayesian_poly_future_forcast00L = bayesian_poly00L.fit_transform(future_forcast)

# polynomial regression
linear_model00L = LinearRegression(normalize=True, fit_intercept=False)
linear_model00L.fit(poly_X_train_deaths00L, y_train_deaths00L)
test_linear_pred00L = linear_model00L.predict(poly_X_test_deaths00L)
linear_pred00L = linear_model00L.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(test_linear_pred00L, y_test_deaths00L))

print(linear_model00L.coef_)


# bayesian ridge polynomial regression
tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
normalize = [True, False]

bayesian_grid00L = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                 'normalize' : normalize}

bayesian00L = BayesianRidge(fit_intercept=False)
bayesian_search00L = RandomizedSearchCV(bayesian00L, bayesian_grid00L, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search00L.fit(bayesian_poly_X_train_deaths00L, y_train_deaths00L)

bayesian_search00L.best_params_

bayesian_deaths = bayesian_search00L.best_estimator_
test_bayesian_pred00L = bayesian_deaths.predict(bayesian_poly_X_test_deaths00L)
bayesian_pred00L = bayesian_deaths.predict(bayesian_poly_future_forcast00L)
print('MAE:', mean_absolute_error(test_bayesian_pred00L, y_test_deaths00L))
print('MSE:',mean_squared_error(test_bayesian_pred00L, y_test_deaths00L))
#50 Recovered
%matplotlib notebook
plt.style.use('seaborn-darkgrid')
plt.figure(figsize=(20, 12))
plt.plot(adjusted_dates, total_recovered, linestyle='solid', color='black')
plt.plot(future_forcast, svm_pred00L, linestyle='dotted', color='purple')
plt.plot(future_forcast, linear_pred00L, linestyle='dotted', color='orange')
plt.plot(future_forcast, bayesian_pred00L, linestyle='dotted', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Recovered Cases', 'SVM predictions', 'Polynomial Regression Predictions', 'Bayesian Ridge Regression Predictions'])
plt.xticks(size=20)
plt.show()