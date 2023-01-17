import pandas as pd 
import numpy as np 

import random
import math
import time
import datetime
import operator 

import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.style.use('seaborn')
%matplotlib inline
infected_ds = pd.read_csv('/kaggle/input/covid-confirmed/time_series_covid19_confirmed_global.csv')
fatalities_ds = pd.read_csv('/kaggle/input/covid-death/time_series_covid19_deaths_global.csv')
recovered_ds = pd.read_csv('/kaggle/input/covid-confirmed/time_series_covid19_confirmed_global.csv')
infected_ds.head(10)
columns = infected_ds.keys()
infected = infected_ds.loc[:, columns[4]:columns[-1]]
fatalities = fatalities_ds.loc[:, columns[4]:columns[-1]]
recovered = recovered_ds.loc[:, columns[4]:columns[-1]]
datelist = infected.keys()

global_count = []
us_count = []
china_count = []
italy_count = []
spain_count = []
india_count = []

total_active = []
total_recovered = []
total_fatalities = []

mortality_rate = []
recovery_rate = []
for x in datelist:
    infected_sum = infected[x].sum()
    fatalities_sum = fatalities[x].sum()
    recovered_sum = recovered[x].sum()
    
    # infetected, fatalities, recovered, and active cases overall
    global_count.append(infected_sum)
    total_fatalities.append(fatalities_sum)
    total_recovered.append(recovered_sum)
    total_active.append(infected_sum-fatalities_sum-recovered_sum)
    
    # calculating mortality and recoery rate overall.
    mortality_rate.append(fatalities_sum/infected_sum)
    recovery_rate.append(recovered_sum/infected_sum)

    # storing infected count country wise (China, Italy, Spain, India)
    us_count.append(infected_ds[infected_ds['Country/Region']=='US'][x].sum())
    china_count.append(infected_ds[infected_ds['Country/Region']=='China'][x].sum())
    italy_count.append(infected_ds[infected_ds['Country/Region']=='Italy'][x].sum())
    spain_count.append(infected_ds[infected_ds['Country/Region']=='Spain'][x].sum())
    india_count.append(infected_ds[infected_ds['Country/Region']=='India'][x].sum())
def increase(ds):
    d = [] 
    for i in range(len(ds)):
        if i == 0:
            d.append(ds[0])
        else:
            d.append(ds[i]-ds[i-1])
    return d 

globalDailyInc = increase(global_count)
usDailyInc = increase(us_count)
chinaDailyInc = increase(china_count)
italyDailyInc = increase(italy_count)
indiaDailyInc = increase(india_count)
spainDailyInc = increase(spain_count)
daysSince22Jan = np.array([i for i in range(len(datelist))]).reshape(-1, 1)
global_count = np.array(global_count).reshape(-1, 1)
total_fatalities = np.array(total_fatalities).reshape(-1, 1)
total_recovered = np.array(total_recovered).reshape(-1, 1)
futureDays = 15
forecast = np.array([i for i in range(len(datelist)+futureDays)]).reshape(-1, 1)
alteredDates = forecast[:-15]
# Number of Covid-19 cases over time
alteredDates = alteredDates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, global_count)
plt.title('No. of Covid-19 Cases Over Time', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#Log of the No. of Covid-19 Cases Over Time
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, np.log10(global_count))
plt.title('Log of the No. of Covid-19 Cases Over Time', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Global Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, globalDailyInc)
plt.title('Global Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# China Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, chinaDailyInc)
plt.title('China Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# US Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, usDailyInc)
plt.title('US Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Italy Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, italyDailyInc)
plt.title('Italy Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Spain Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, spainDailyInc)
plt.title('Spain Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# India Daily Increases in Infected Count (Tested Positive)
plt.figure(figsize=(16, 9))
plt.bar(alteredDates, indiaDailyInc)
plt.title('India Daily Increases in Infected Count (Tested Positive)', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Comparison between countries
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, china_count)
plt.plot(alteredDates, us_count)
plt.plot(alteredDates, italy_count)
plt.plot(alteredDates, spain_count)
plt.plot(alteredDates, india_count)
plt.title('No. of Covid-19 Cases', size=30)
plt.xlabel('Days Since 22 Jan, 2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.legend(['China','US', 'Italy', 'Spain', 'India'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
begin = '1/22/2020'
initDate = datetime.datetime.strptime(begin, '%m/%d/%Y')
forecastDates = []

for i in range(len(forecast)):
    forecastDates.append((initDate + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
XTrainInfected, XTestInfected, yTrainInfected, yTestInfected = train_test_split(daysSince22Jan, global_count, test_size=0.15, 
                                                                                shuffle=False)
# Linear Regression
# Transforming Data
plyReg = PolynomialFeatures(degree=5)
plyRegXTrainInfected = plyReg.fit_transform(XTrainInfected)
plyRegXTestInfected = plyReg.fit_transform(XTestInfected)
plyRegForecast = plyReg.fit_transform(forecast)
# Training
lrModel = LinearRegression(normalize=True, fit_intercept=False)
lrModel.fit(plyRegXTrainInfected, yTrainInfected)
lrFdatesPredict = lrModel.predict(plyRegForecast)
# Testing
lrTestPredict = lrModel.predict(plyRegXTestInfected)
plt.plot(yTestInfected)
plt.plot(lrTestPredict)
plt.legend(['Testing Data', 'Linear Regression Predictions'])
print('MAE:', mean_absolute_error(lrTestPredict, yTestInfected))
print('MSE:',mean_squared_error(lrTestPredict, yTestInfected))
print('Coefficients:',lrModel.coef_)
# Support Vector Machine
# Training
svmInfected = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1)
svmInfected.fit(XTrainInfected, yTrainInfected)
svmFdatesPredict = svmInfected.predict(forecast)
# Testing
svmTestPredict = svmInfected.predict(XTestInfected)
plt.plot(yTestInfected)
plt.plot(svmTestPredict)
plt.legend(['Testing Data', 'SVM Predictions'])
print('MAE:', mean_absolute_error(svmTestPredict, yTestInfected))
print('MSE:',mean_squared_error(svmTestPredict, yTestInfected))
# Bayesian Ridge Polynomial Regression
# Training
# Searching for the optimal parameters to use for the Baysian Ridge Forecast using RandomizedSearchCV 
# and Training the model.
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesGrid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayes = BayesianRidge(fit_intercept=False, normalize=True)
bayesSearch = RandomizedSearchCV(bayes, bayesGrid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesSearch.fit(plyRegXTrainInfected, yTrainInfected)
# The optimal parameters
bayesSearch.best_params_
bayesInfected = bayesSearch.best_estimator_
bayesFdatesPredict = bayesInfected.predict(plyRegForecast)
# Testing
bayesTestPredict = bayesInfected.predict(plyRegXTestInfected)
plt.plot(yTestInfected)
plt.plot(bayesTestPredict)
plt.legend(['Testing Data', 'Bayesian Ridge Predictions'])
print('MAE:', mean_absolute_error(bayesTestPredict, yTestInfected))
print('MSE:',mean_squared_error(bayesTestPredict, yTestInfected))
# Forecast using Linear Regression 
lrFdatesPredict = lrFdatesPredict.reshape(1,-1)[0]
print('Linear regression future predictions:')
set(zip(forecastDates[-15:], np.round(lrFdatesPredict[-15:])))
# Linear Regression Forecast Plot
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, global_count)
plt.plot(forecast, lrFdatesPredict, linestyle='dashed', color='red')
plt.title('No. of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.legend(['Confirmed Cases', 'Linear Regression Forecast'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Forecast using SVM
print('SVM future predictions:')
set(zip(forecastDates[-15:], np.round(svmFdatesPredict[-15:])))
# SVM Forecast Plot
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, global_count)
plt.plot(forecast, svmFdatesPredict, linestyle='dashed', color='red')
plt.title('No. of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('No. of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM Forecast'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Forecast using Bayesian Ridge 
print('Bayesian Ridge future predictions:')
set(zip(forecastDates[-15:], np.round(bayesFdatesPredict[-15:])))
# Bayesian Ridge Forecast Plot
plt.figure(figsize=(16, 9))
plt.plot(alteredDates, global_count)
plt.plot(forecast, bayesFdatesPredict, linestyle='dashed', color='red')
plt.title('No. of Coronavirus Cases Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('No. of Cases', size=30)
plt.legend(['Confirmed Cases', 'Bayesian Ridge Forecast'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()