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
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator 
plt.style.use('fivethirtyeight')
%matplotlib inline 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
confirmed_df = pd.read_csv("/kaggle/input/sweden-covid19-dataset/time_series_confimed-confirmed.csv")
deaths_df = pd.read_csv('/kaggle/input/sweden-covid19-dataset/time_series_deaths-deaths.csv')

confirmed_df.head(50)
deaths_df.head(30)
cols = confirmed_df.keys()
confirmed = confirmed_df.loc[:21, cols[5]:cols[55]]
deaths = deaths_df.loc[:21, cols[5]:cols[55]]


confirmed.sum().sum()
deaths.sum().sum()
cols = confirmed_df.keys()
cols

confirmed_df
dates = confirmed.keys()
sweden_cases = []
total_deaths = [] 
mortality_rate = []


        
Blekinge_cases=[]
Dalarna_cases=[]
Gotland_cases=[]
Gavleborg_cases=[]
Halland_cases=[]
Jamtland_cases=[]
Jonkoping_cases=[]
Kalmarlan_cases=[]
Kronoberg_cases=[]
Norrbotten_cases=[]
Skane_cases=[]
Stockholm_cases=[]
Sormland_cases=[]
Uppsala_cases=[]
Varmland_cases=[]
Vasterbotten_cases=[]
Vasternorrland_cases=[]
Vastmanland_cases=[]
VastraGotaland_cases=[]
Orebro_cases=[]
Ostergotland_cases=[]
Okant_cases=[]


Blekinge_deaths=[]
Dalarna_deaths=[]
Gotland_deaths=[]
Gavleborg_deaths=[]
Halland_deaths=[]
Jamtland_deaths=[]
Jonkoping_deaths=[]
Kalmarlan_deaths=[]
Kronoberg_deaths=[]
Norrbotten_deaths=[]
Skane_deaths=[]
Stockholm_deaths=[]
Sormlan_deaths=[]
Uppsala_deaths=[]
Varmland_deaths=[]
Vasterbotten_deaths=[]
Vasternorrland_deaths=[]
Vastmanland_deaths=[]
VastraGotaland_deaths=[]
Orebro_deaths=[]
Ostergotland_deaths=[]
Okant_deaths=[]

for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    
    # confirmed, deaths, recovered, and active
    sweden_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    
    
    # calculate rates
    mortality_rate.append(death_sum/confirmed_sum)
    
    

    # case studies 
 
    Blekinge_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Blekinge"][i].sum())
    Dalarna_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Dalarna"][i].sum())
    Gotland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Gotland"][i].sum())
    Gavleborg_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Gävleborg"][i].sum())
    Halland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Halland"][i].sum())
    Jamtland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Jämtland"][i].sum())
    Jonkoping_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Jönköping"][i].sum())
    Kalmarlan_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Kalmar län"][i].sum())
    Kronoberg_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Kronoberg"][i].sum())
    Norrbotten_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Norrbotten"][i].sum())
    Skane_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Skåne"][i].sum())
    Stockholm_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Stockholm"][i].sum())
    Sormland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Sörmland"][i].sum())
    Uppsala_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Uppsala"][i].sum())
    Varmland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Värmland"][i].sum())
    Vasterbotten_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Västerbotten"][i].sum())
    Vasternorrland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Västernorrland"][i].sum())
    Vastmanland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Västmanland"][i].sum())
    VastraGotaland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Västra Götaland"][i].sum())
    Orebro_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Örebro"][i].sum())
    Ostergotland_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Östergötland"][i].sum())
    Okant_cases.append(confirmed_df[confirmed_df["Display_Name"]=="Okänt"][i].sum())
     
    # deaths studies 
    Blekinge_deaths.append(deaths_df[deaths_df["Display_Name"]=="Blekinge"][i].sum())
    Dalarna_deaths.append(deaths_df[deaths_df["Display_Name"]=="Dalarna"][i].sum())
    Gotland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Gotland"][i].sum())
    Gavleborg_deaths.append(deaths_df[deaths_df["Display_Name"]=="Gävleborg"][i].sum())
    Halland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Halland"][i].sum())
    Jamtland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Jämtland"][i].sum())
    Jonkoping_deaths.append(deaths_df[deaths_df["Display_Name"]=="Jönköping"][i].sum())
    Kalmarlan_deaths.append(deaths_df[deaths_df["Display_Name"]=="Kalmar län"][i].sum())
    Kronoberg_deaths.append(deaths_df[deaths_df["Display_Name"]=="Kronoberg"][i].sum())
    Norrbotten_deaths.append(deaths_df[deaths_df["Display_Name"]=="Norrbotten"][i].sum())
    Skane_deaths.append(deaths_df[deaths_df["Display_Name"]=="Skåne"][i].sum())
    Stockholm_deaths.append(deaths_df[deaths_df["Display_Name"]=="Stockholm"][i].sum())
    Sormlan_deaths.append(deaths_df[deaths_df["Display_Name"]=="Sörmland"][i].sum())
    Uppsala_deaths.append(deaths_df[deaths_df["Display_Name"]=="Uppsala"][i].sum())
    Varmland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Värmland"][i].sum())
    Vasterbotten_deaths.append(deaths_df[deaths_df["Display_Name"]=="Västerbotten"][i].sum())
    Vasternorrland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Västernorrland"][i].sum())
    Vastmanland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Västmanland"][i].sum())
    VastraGotaland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Västra Götaland"][i].sum())
    Orebro_deaths.append(deaths_df[deaths_df["Display_Name"]=="Örebro"][i].sum())
    Ostergotland_deaths.append(deaths_df[deaths_df["Display_Name"]=="Östergötland"][i].sum())
    Okant_deaths.append(deaths_df[deaths_df["Display_Name"]=="Okänt"][i].sum())
    
    

def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

# confirmed cases
sweden_daily_increase = daily_increase(sweden_cases)


# deaths
sweden_daily_death = daily_increase(total_deaths)



days_since_2_25 = np.array([i for i in range(len(dates))]).reshape(-1, 1)
sweden_cases = np.array(sweden_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)

days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '2/25/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days_since_2_25, sweden_cases, test_size=0.10, shuffle=False) 
# use this to find the optimal parameters for SVR
# c = [0.01, 0.1, 1]
# gamma = [0.01, 0.1, 1]
# epsilon = [0.01, 0.1, 1]
# shrinking = [True, False]
# degree = [3, 4, 5, 6, 7]

# svm_grid = {'C': c, 'gamma' : gamma, 'epsilon': epsilon, 'shrinking' : shrinking, 'degree': degree}

# svm = SVR(kernel='poly')
# svm_search = RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=30, verbose=1)
# svm_search.fit(X_train_confirmed, y_train_confirmed)
# svm_search.best_params_
# svm_confirmed = svm_search.best_estimator_
svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=5, C=0.1)
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
poly = PolynomialFeatures()
poly_X_train_confirmed = poly.fit_transform(X_train_confirmed)
poly_X_test_confirmed = poly.fit_transform(X_test_confirmed)
poly_future_forcast = poly.fit_transform(future_forcast)

bayesian_poly = PolynomialFeatures(degree=1)
bayesian_poly_X_train_confirmed = bayesian_poly.fit_transform(X_train_confirmed)
bayesian_poly_X_test_confirmed = bayesian_poly.fit_transform(X_test_confirmed)
bayesian_poly_future_forcast = bayesian_poly.fit_transform(future_forcast)
# polynomial regression
linear_model = LinearRegression(normalize=True, fit_intercept=True)
linear_model.fit(poly_X_train_confirmed, y_train_confirmed)
test_linear_pred = linear_model.predict(poly_X_test_confirmed)
linear_pred = linear_model.predict(poly_future_forcast)
print('MAE:', mean_absolute_error(test_linear_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_linear_pred, y_test_confirmed))
print(linear_model.coef_)
plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search.fit(bayesian_poly_X_train_confirmed, y_train_confirmed)
bayesian_search.best_params_
bayesian_confirmed = bayesian_search.best_estimator_
test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test_confirmed)
bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forcast)
print('MAE:', mean_absolute_error(test_bayesian_pred, y_test_confirmed))
print('MSE:',mean_squared_error(test_bayesian_pred, y_test_confirmed))
plt.plot(y_test_confirmed)
plt.plot(test_bayesian_pred)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, sweden_cases)
plt.title(' Coronavirus Cases Over Time in Sweden', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=30)
plt.yticks(size=30)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, total_deaths)
plt.title('# of Coronavirus Deaths Over Time', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()


adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Stockholm_cases)
plt.title(' Coronavirus Cases Over Time in Stockholm', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('Stockholm of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, Stockholm_deaths)
plt.title('Coronavirus Deaths Over Time in Stockholm', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, VastraGotaland_cases)
plt.title(' Coronavirus Cases Over Time in Västra Götaland', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('Stockholm of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, VastraGotaland_deaths)
plt.title('Coronavirus Deaths Over Time in Västra Götaland', size=30)
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
def plot_predictions(x, y, pred, algo_name, color):
    plt.figure(figsize=(16, 9))
    plt.plot(x, y)
    plt.plot(future_forcast, pred, linestyle='dashed', color=color)
    plt.title(' Coronavirus Cases Over Time in Sweden', size=30)
    plt.xlabel('Days Since 2/25/2020', size=30)
    plt.ylabel('# of Cases', size=30)
    plt.legend(['Confirmed Cases', algo_name], prop={'size': 20})
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.show()
plot_predictions(adjusted_dates, sweden_cases, svm_pred, 'SVM Predictions', 'purple')
plot_predictions(adjusted_dates, sweden_cases, linear_pred, 'Polynomial Regression Predictions', 'orange')
plot_predictions(adjusted_dates, sweden_cases, bayesian_pred, 'Bayesian Ridge Regression Predictions', 'green')
# Future predictions using SVM 
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'SVM Predicted # of Confirmed Cases in Sweden': np.round(svm_pred[-10:])})
svm_df
# Future predictions using polynomial regression
linear_pred = linear_pred.reshape(1,-1)[0]
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Polynomial Predicted # of Confirmed Cases in Sweden': np.round(linear_pred[-10:])})
svm_df
# Future predictions using Bayesian Ridge 
svm_df = pd.DataFrame({'Date': future_forcast_dates[-10:], 'Bayesian Ridge Predicted # of Confirmed Cases in Sweden': np.round(bayesian_pred[-10:])})
svm_df
mean_mortality_rate = np.mean(mortality_rate)
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, mortality_rate, color='orange')
#plt.axhline(y= ,linestyle='--', color='black')
plt.title('Mortality Rate of Coronavirus Over Time', size=30)
plt.legend(['mortality rate', 'y='+str(0.0367)], prop={'size': 20})
plt.xlabel('Days Since 2/25/2020', size=30)
plt.ylabel('Mortality Rate', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()



