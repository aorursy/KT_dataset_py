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
import seaborn as sns
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
latest_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/03-31-2020.csv')



confirmed_df.head()
cols = confirmed_df.keys()

confirmed = confirmed_df.loc[:, cols[4]:cols[-1]]
deaths = deaths_df.loc[:, cols[4]:cols[-1]]
recoveries = recoveries_df.loc[:, cols[4]:cols[-1]]

confirmed_arg = confirmed_df.loc[6,cols[4]:cols[-1]]
deaths_arg= deaths_df.loc[6,cols[4]:cols[-1]]
recoveries_arg = recoveries_df.loc[6,cols[4]:cols[-1]]

confirmed_bra = confirmed_df.loc[28,cols[4]:cols[-1]]
deaths_bra= deaths_df.loc[28,cols[4]:cols[-1]]
recoveries_bra = recoveries_df.loc[28,cols[4]:cols[-1]]

confirmed_bol = confirmed_df.loc[26,cols[4]:cols[-1]]
deaths_bol= deaths_df.loc[26,cols[4]:cols[-1]]
recoveries_bol = recoveries_df.loc[26,cols[4]:cols[-1]]

confirmed_chi = confirmed_df.loc[48,cols[4]:cols[-1]]
deaths_chi= deaths_df.loc[48,cols[4]:cols[-1]]
recoveries_chi = recoveries_df.loc[48,cols[4]:cols[-1]]

confirmed_par = confirmed_df.loc[180,cols[4]:cols[-1]]
deaths_par= deaths_df.loc[180,cols[4]:cols[-1]]
recoveries_par = recoveries_df.loc[180,cols[4]:cols[-1]]

confirmed_peru = confirmed_df.loc[181,cols[4]:cols[-1]]
deaths_peru= deaths_df.loc[181,cols[4]:cols[-1]]
recoveries_peru = recoveries_df.loc[181,cols[4]:cols[-1]]

confirmed_chi = confirmed_df.loc[48,cols[4]:cols[-1]]
deaths_chi= deaths_df.loc[48,cols[4]:cols[-1]]
recoveries_chi = recoveries_df.loc[48,cols[4]:cols[-1]]

confirmed_ven = confirmed_df.loc[227,cols[4]:cols[-1]]
deaths_ven= deaths_df.loc[227,cols[4]:cols[-1]]
recoveries_ven = recoveries_df.loc[227,cols[4]:cols[-1]]

confirmed_uru = confirmed_df.loc[224,cols[4]:cols[-1]]
deaths_uru= deaths_df.loc[224,cols[4]:cols[-1]]
recoveries_uru = recoveries_df.loc[224,cols[4]:cols[-1]]

confirmed_col = confirmed_df.loc[82,cols[4]:cols[-1]]
deaths_col= deaths_df.loc[82,cols[4]:cols[-1]]
recoveries_col = recoveries_df.loc[82,cols[4]:cols[-1]]

confirmed_ecu = confirmed_df.loc[97,cols[4]:cols[-1]]
deaths_ecu= deaths_df.loc[97,cols[4]:cols[-1]]
recoveries_ecu = recoveries_df.loc[97,cols[4]:cols[-1]]
dates = confirmed.keys()

#MUNDIAL

world_cases = []
total_deaths = [] 
total_recovered = []
mortality_rate = []
recovery_rate = [] 

#ARGENTINA

total_active = [] 
argentina_cases = []
argentina_deaths = []
argentina_recovered = []
arg_mortality_rate = []
arg_recovery_rate = []
#BRAZIL
bra_cases = []
bra_deaths = []
bra_recovered = []
bra_mortality_rate = []
bra_recovery_rate = []
#BOLIVIA
bol_cases = []
bol_deaths = []
bol_recovered = []
bol_mortality_rate = []
bol_recovery_rate = []
#CHILE
chi_cases = []
chi_deaths = []
chi_recovered = []
chi_mortality_rate = []
chi_recovery_rate = []
#PARAGUAY
par_cases = []
par_deaths = []
par_recovered = []
par_mortality_rate = []
par_recovery_rate = []
#PERU
peru_cases = []
peru_deaths = []
peru_recovered = []
peru_mortality_rate = []
peru_recovery_rate = []
#VENEZUELA
ven_cases = []
ven_deaths = []
ven_recovered = []
ven_mortality_rate = []
ven_recovery_rate = []
#URUGUAY
uru_cases = []
uru_deaths = []
uru_recovered = []
uru_mortality_rate = []
uru_recovery_rate = []
#COLOMBIA
col_cases = []
col_deaths = []
col_recovered = []
col_mortality_rate = []
col_recovery_rate = []
#ECUADOR
ecu_cases = []
ecu_deaths = []
ecu_recovered = []
ecu_mortality_rate = []
ecu_recovery_rate = []




for i in dates:
    confirmed_sum = confirmed[i].sum()
    death_sum = deaths[i].sum()
    recoveries_sum = recoveries[i].sum()
    
    arg_confirmed_sum = confirmed_arg[i].sum()
    arg_death_sum = deaths_arg[i].sum()
    recov_arg_sum = recoveries_arg[i].sum()
    
    bra_confirmed_sum = confirmed_bra[i].sum()
    bra_death_sum = deaths_bra[i].sum()
    recov_bra_sum = recoveries_bra[i].sum()
    
    bol_confirmed_sum = confirmed_bol[i].sum()
    bol_death_sum = deaths_bol[i].sum()
    recov_bol_sum = recoveries_bol[i].sum()
    
    chi_confirmed_sum = confirmed_chi[i].sum()
    chi_death_sum = deaths_chi[i].sum()
    recov_chi_sum = recoveries_chi[i].sum()
    
    par_confirmed_sum = confirmed_par[i].sum()
    par_death_sum = deaths_par[i].sum()
    recov_par_sum = recoveries_par[i].sum()
    
    peru_confirmed_sum = confirmed_peru[i].sum()
    peru_death_sum = deaths_peru[i].sum()
    recov_peru_sum = recoveries_peru[i].sum()
    
    ven_confirmed_sum = confirmed_ven[i].sum()
    ven_death_sum = deaths_ven[i].sum()
    recov_ven_sum = recoveries_ven[i].sum()
    
    uru_confirmed_sum = confirmed_uru[i].sum()
    uru_death_sum = deaths_uru[i].sum()
    recov_uru_sum = recoveries_uru[i].sum()
    
    col_confirmed_sum = confirmed_col[i].sum()
    col_death_sum = deaths_col[i].sum()
    recov_col_sum = recoveries_col[i].sum()
    
    ecu_confirmed_sum = confirmed_ecu[i].sum()
    ecu_death_sum = deaths_ecu[i].sum()
    recov_ecu_sum = recoveries_ecu[i].sum()

    
    # MUNDIAL
    world_cases.append(confirmed_sum)
    total_deaths.append(death_sum)
    total_recovered.append(recoveries_sum)
    mortality_rate.append(death_sum/confirmed_sum)
    recovery_rate.append(recoveries_sum/confirmed_sum)
    
    #ARGENTINA
    argentina_cases.append(confirmed_df[confirmed_df['Country/Region']=='Argentina'][i].sum())
    argentina_deaths.append(deaths_df[deaths_df['Country/Region']=='Argentina'][i].sum())
    arg_mortality_rate.append(arg_death_sum/arg_confirmed_sum)
    argentina_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Argentina'][i].sum())
    arg_recovery_rate.append(recov_arg_sum/arg_confirmed_sum)
    
    #BRASIL
    bra_cases.append(confirmed_df[confirmed_df['Country/Region']=='Brazil'][i].sum())
    bra_deaths.append(deaths_df[deaths_df['Country/Region']=='Brazil'][i].sum())
    bra_mortality_rate.append(bra_death_sum/bra_confirmed_sum)
    bra_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Brazil'][i].sum())
    bra_recovery_rate.append(recov_bra_sum/bra_confirmed_sum)
    
    #BOLIVIA
    bol_cases.append(confirmed_df[confirmed_df['Country/Region']=='Bolivia'][i].sum())
    bol_deaths.append(deaths_df[deaths_df['Country/Region']=='Bolivia'][i].sum())
    bol_mortality_rate.append(bol_death_sum/bol_confirmed_sum)
    bol_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Bolivia'][i].sum())
    bol_recovery_rate.append(recov_bol_sum/bol_confirmed_sum)
    
    #CHILE
    chi_cases.append(confirmed_df[confirmed_df['Country/Region']=='Chile'][i].sum())
    chi_deaths.append(deaths_df[deaths_df['Country/Region']=='Chile'][i].sum())
    chi_mortality_rate.append(chi_death_sum/chi_confirmed_sum)
    chi_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Chile'][i].sum())
    chi_recovery_rate.append(recov_chi_sum/chi_confirmed_sum)  

    #PARAGUAY
    par_cases.append(confirmed_df[confirmed_df['Country/Region']=='Paraguay'][i].sum())
    par_deaths.append(deaths_df[deaths_df['Country/Region']=='Paraguay'][i].sum())
    par_mortality_rate.append(par_death_sum/par_confirmed_sum)
    par_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Paraguay'][i].sum())
    par_recovery_rate.append(recov_par_sum/par_confirmed_sum)
    
    #PERU
    peru_cases.append(confirmed_df[confirmed_df['Country/Region']=='Peru'][i].sum())
    peru_deaths.append(deaths_df[deaths_df['Country/Region']=='Peru'][i].sum())
    peru_mortality_rate.append(peru_death_sum/peru_confirmed_sum)
    peru_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Peru'][i].sum())
    peru_recovery_rate.append(recov_peru_sum/peru_confirmed_sum)
    
    #VENEZUELA
    ven_cases.append(confirmed_df[confirmed_df['Country/Region']=='Venezuela'][i].sum())
    ven_deaths.append(deaths_df[deaths_df['Country/Region']=='Venezuela'][i].sum())
    ven_mortality_rate.append(ven_death_sum/ven_confirmed_sum)
    ven_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Venezuela'][i].sum())
    ven_recovery_rate.append(recov_ven_sum/ven_confirmed_sum)
    
    #URUGUAY
    uru_cases.append(confirmed_df[confirmed_df['Country/Region']=='Uruguay'][i].sum())
    uru_deaths.append(deaths_df[deaths_df['Country/Region']=='Uruguay'][i].sum())
    uru_mortality_rate.append(uru_death_sum/uru_confirmed_sum)
    uru_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Uruguay'][i].sum())
    uru_recovery_rate.append(recov_uru_sum/uru_confirmed_sum)
    
    #COLOMBIA
    col_cases.append(confirmed_df[confirmed_df['Country/Region']=='Colombia'][i].sum())
    col_deaths.append(deaths_df[deaths_df['Country/Region']=='Colombia'][i].sum())
    col_mortality_rate.append(col_death_sum/col_confirmed_sum)
    col_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Colombia'][i].sum())
    col_recovery_rate.append(recov_col_sum/col_confirmed_sum)
    
    #ECUADOR
    ecu_cases.append(confirmed_df[confirmed_df['Country/Region']=='Ecuador'][i].sum())
    ecu_deaths.append(deaths_df[deaths_df['Country/Region']=='Ecuador'][i].sum())
    ecu_mortality_rate.append(ecu_death_sum/ecu_confirmed_sum)
    ecu_recovered.append(recoveries_df[recoveries_df['Country/Region']=='Ecuador'][i].sum())
    ecu_recovery_rate.append(recov_ecu_sum/ecu_confirmed_sum)
def daily_increase(data):
    d = [] 
    for i in range(len(data)):
        if i == 0:
            d.append(data[0])
        else:
            d.append(data[i]-data[i-1])
    return d 

world_daily_increase = daily_increase(world_cases)

argentina_daily_increase = daily_increase(argentina_cases)
argentina_daily_increase_deaths = daily_increase(argentina_deaths)
argentina_daily_increase_recovered = daily_increase(argentina_recovered)

bra_daily_increase = daily_increase(bra_cases)
bra_daily_increase_deaths = daily_increase(bra_deaths)
bra_daily_increase_recovered = daily_increase(bra_recovered)

bol_daily_increase = daily_increase(bol_cases)
bol_daily_increase_deaths = daily_increase(bol_deaths)
bol_daily_increase_recovered = daily_increase(bol_recovered)

chi_daily_increase = daily_increase(chi_cases)
chi_daily_increase_deaths = daily_increase(chi_deaths)
chi_daily_increase_recovered = daily_increase(chi_recovered)

par_daily_increase = daily_increase(par_cases)
par_daily_increase_deaths = daily_increase(par_deaths)
par_daily_increase_recovered = daily_increase(par_recovered)

peru_daily_increase = daily_increase(peru_cases)
peru_daily_increase_deaths = daily_increase(peru_deaths)
peru_daily_increase_recovered = daily_increase(peru_recovered)

ven_daily_increase = daily_increase(ven_cases)
ven_daily_increase_deaths = daily_increase(ven_deaths)
ven_daily_increase_recovered = daily_increase(ven_recovered)

uru_daily_increase = daily_increase(uru_cases)
uru_daily_increase_deaths = daily_increase(uru_deaths)
uru_daily_increase_recovered = daily_increase(uru_recovered)

col_daily_increase = daily_increase(col_cases)
col_daily_increase_deaths = daily_increase(col_deaths)
col_daily_increase_recovered = daily_increase(col_recovered)

ecu_daily_increase = daily_increase(ecu_cases)
ecu_daily_increase_deaths = daily_increase(ecu_deaths)
ecu_daily_increase_recovered = daily_increase(ecu_recovered)


days_since_1_22 = np.array([i for i in range(len(dates))]).reshape(-1, 1)

world_cases = np.array(world_cases).reshape(-1, 1)
total_deaths = np.array(total_deaths).reshape(-1, 1)

argentina_cases = np.array(argentina_cases).reshape(-1, 1)
argentina_deaths = np.array(argentina_deaths).reshape(-1, 1)
argentina_recovered = np.array(argentina_recovered).reshape(-1,1)

bol_cases = np.array(bol_cases).reshape(-1, 1)
bol_deaths = np.array(bol_deaths).reshape(-1, 1)
bol_recovered = np.array(bol_recovered).reshape(-1,1)

bra_cases = np.array(bra_cases).reshape(-1, 1)
bra_deaths = np.array(bra_deaths).reshape(-1, 1)
bra_recovered = np.array(bra_recovered).reshape(-1,1)

chi_cases = np.array(chi_cases).reshape(-1, 1)
chi_deaths = np.array(chi_deaths).reshape(-1, 1)
chi_recovered = np.array(chi_recovered).reshape(-1,1)

par_cases = np.array(par_cases).reshape(-1, 1)
par_deaths = np.array(par_deaths).reshape(-1, 1)
par_recovered = np.array(par_recovered).reshape(-1,1)

peru_cases = np.array(peru_cases).reshape(-1, 1)
peru_deaths = np.array(peru_deaths).reshape(-1, 1)
peru_recovered = np.array(peru_recovered).reshape(-1,1)

ven_cases = np.array(ven_cases).reshape(-1, 1)
ven_deaths = np.array(ven_deaths).reshape(-1, 1)
ven_recovered = np.array(ven_recovered).reshape(-1,1)

uru_cases = np.array(uru_cases).reshape(-1, 1)
uru_deaths = np.array(uru_deaths).reshape(-1, 1)
uru_recovered = np.array(uru_recovered).reshape(-1,1)

col_cases = np.array(col_cases).reshape(-1, 1)
col_deaths = np.array(col_deaths).reshape(-1, 1)
col_recovered = np.array(col_recovered).reshape(-1,1)

ecu_cases = np.array(ecu_cases).reshape(-1, 1)
ecu_deaths = np.array(ecu_deaths).reshape(-1, 1)
ecu_recovered = np.array(ecu_recovered).reshape(-1,1)


days_in_future = 10
future_forcast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1, 1)
adjusted_dates = future_forcast[:-10]
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forcast_dates = []
for i in range(len(future_forcast)):
    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))
X_train_confirmed_arg, X_test_confirmed_arg, y_train_confirmed_arg, y_test_confirmed_arg = train_test_split(days_since_1_22, argentina_cases, test_size=0.15, shuffle=False)
svm_confirmed_arg = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1,degree=6, C=0.1)
svm_confirmed_arg.fit(X_train_confirmed_arg, y_train_confirmed_arg)
svm_pred_arg = svm_confirmed_arg.predict(future_forcast)
# check against testing data
svm_test_pred_arg = svm_confirmed_arg.predict(X_test_confirmed_arg)
plt.plot(svm_test_pred_arg)
plt.plot(y_test_confirmed_arg)
print('MAE:', mean_absolute_error(svm_test_pred_arg, y_test_confirmed_arg))
print('MSE:',mean_squared_error(svm_test_pred_arg, y_test_confirmed_arg))

plt.legend(['Test Data', 'SVM Predictions'])
#AJUSTO FECHAS ULTIMOS 30 DIAS


adjusted_dates_america=adjusted_dates[-30:]

argentina_daily_increase_ultimos_30_dias= argentina_daily_increase[-30:]
argentina_daily_increase_deaths_ultimos_30_dias = argentina_daily_increase_deaths[-30:]

argentina_cases_ult_30_dias = argentina_cases[-30:]
plt.figure(figsize=(12, 6))
plt.plot(adjusted_dates, argentina_cases)
plt.plot(future_forcast, svm_pred_arg, linestyle='dashed', color='purple')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'SVM predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Future predictions using SVM 
print('SVM future predictions:')
set(zip(future_forcast_dates[-10:], np.round(svm_pred_arg[-10:])))
# transform our data for polynomial regression
poly_arg = PolynomialFeatures(degree=5)
poly_X_train_confirmed_arg = poly_arg.fit_transform(X_train_confirmed_arg)
poly_X_test_confirmed_arg = poly_arg.fit_transform(X_test_confirmed_arg)
poly_future_forcast_arg = poly_arg.fit_transform(future_forcast)
# polynomial regression
linear_model_arg = LinearRegression(normalize=True, fit_intercept=False)
linear_model_arg.fit(poly_X_train_confirmed_arg, y_train_confirmed_arg)
test_linear_pred_arg = linear_model_arg.predict(poly_X_test_confirmed_arg)
linear_pred_arg = linear_model_arg.predict(poly_future_forcast_arg)
print('MAE:', mean_absolute_error(test_linear_pred_arg, y_test_confirmed_arg))
print('MSE:',mean_squared_error(test_linear_pred_arg, y_test_confirmed_arg))
print(linear_model_arg.coef_)
plt.plot(test_linear_pred_arg)
plt.plot(y_test_confirmed_arg)
plt.legend(['Test Data', 'Polynomial Regression Predictions'])
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates, argentina_cases)
plt.plot(future_forcast, linear_pred_arg, linestyle='dashed', color='orange')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Days Since 1/22/2020', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Polynomial Regression Predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Future predictions using Polynomial Regression 
linear_pred_arg = linear_pred_arg.reshape(1,-1)[0]
print('Polynomial regression future predictions:')
set(zip(future_forcast_dates[-10:], np.round(linear_pred_arg[-10:])))
# bayesian ridge polynomial regression
tol = [1e-4, 1e-3, 1e-2]
alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4]
alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4]
lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4]

bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2}

bayesian_arg = BayesianRidge(fit_intercept=False, normalize=True)
bayesian_search_arg = RandomizedSearchCV(bayesian_arg, bayesian_grid, scoring='neg_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=40, verbose=1)
bayesian_search_arg.fit(poly_X_train_confirmed_arg, y_train_confirmed_arg)
bayesian_search_arg.best_params_
bayesian_confirmed_arg = bayesian_search_arg.best_estimator_
test_bayesian_pred_arg = bayesian_confirmed_arg.predict(poly_X_test_confirmed_arg)
bayesian_pred_arg = bayesian_confirmed_arg.predict(poly_future_forcast_arg)
print('MAE:', mean_absolute_error(test_bayesian_pred_arg, y_test_confirmed_arg))
print('MSE:',mean_squared_error(test_bayesian_pred_arg, y_test_confirmed_arg))
plt.plot(y_test_confirmed_arg)
plt.plot(test_bayesian_pred_arg)
plt.legend(['Test Data', 'Bayesian Ridge Polynomial Predictions'])
plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates_america, argentina_cases_ult_30_dias)
plt.plot(future_forcast, bayesian_pred_arg, linestyle='dashed', color='green')
plt.title('# of Coronavirus Cases Over Time', size=30)
plt.xlabel('Time', size=30)
plt.ylabel('# of Cases', size=30)
plt.legend(['Confirmed Cases', 'Polynomial Bayesian Ridge Regression Predictions'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
# Future predictions using Linear Regression 
print('Ridge regression future predictions:')
set(zip(future_forcast_dates[-10:], np.round(bayesian_pred_arg[-10:])))
argentina_cases_ult_30_dias = argentina_cases[-30:]

adjusted_dates = adjusted_dates.reshape(1, -1)[0]


plt.figure(figsize=(14, 9))
plt.plot(adjusted_dates_america, argentina_cases_ult_30_dias,'b',lw=5)
plt.title('# Evolucion Casos Confirmados Covid-19 en Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=30)
plt.yticks(size=30)
plt.show()


argentina_deaths_ult_30_dias = argentina_deaths[-30:]


adjusted_dates = adjusted_dates.reshape(1, -1)[0]
plt.figure(figsize=(14, 10))
plt.plot(adjusted_dates_america, argentina_deaths_ult_30_dias,'r',lw=5)
plt.title('# Evolucion Fallecimientos Covid-19 en Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)

plt.ylabel('# of Cases', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
argentina_recovered_ult_30_dias = argentina_recovered[-30:]

adjusted_dates = adjusted_dates.reshape(1, -1)[0]


plt.figure(figsize=(14, 9))
plt.plot(adjusted_dates_america, argentina_recovered_ult_30_dias,'g',lw=5)
plt.title('# Evolucion Casos Recuperacion Covid-19 en Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# of Cases', size=30)
plt.xticks(size=30)
plt.yticks(size=30)
plt.show()

plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates_america, argentina_cases_ult_30_dias, lw = 5)
plt.plot(adjusted_dates_america, argentina_deaths_ult_30_dias, lw=5)
plt.plot(adjusted_dates_america,argentina_recovered_ult_30_dias,'g',lw =5)

plt.title('Datos COVID-19 Casos Confirmados - Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# of Casos', size=30)
plt.legend(['Casos Confirmados', 'Fallecimientos', 'Recuperaciones'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#AJUSTO FECHAS ULTIMOS 30 DIAS


adjusted_dates_america=adjusted_dates[-30:]

argentina_daily_increase_ultimos_30_dias= argentina_daily_increase[-30:]
argentina_daily_increase_deaths_ultimos_30_dias = argentina_daily_increase_deaths[-30:]

adjusted_dates_america_test_15_dias = adjusted_dates[-15:]
plt.figure(figsize=(16, 9))
sns.barplot(adjusted_dates_america, argentina_daily_increase_ultimos_30_dias,color = 'forestgreen')
plt.title('Casos Diarios COVID-19 Confirmados Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# de Casos', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()

plt.figure(figsize=(16, 9))
sns.barplot(adjusted_dates_america, argentina_daily_increase_deaths_ultimos_30_dias,color = 'darkred')
plt.title('Casos Diarios Fallecimientos COVID-19 Argentina', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# de Casos', size=30)
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
bra_cases_ult_30_dias = bra_cases[-30:]
bol_cases_ult_30_dias = bol_cases[-30:]
chi_cases_ult_30_dias = chi_cases[-30:]
par_cases_ult_30_dias = par_cases[-30:]
peru_cases_ult_30_dias = peru_cases[-30:]
ven_cases_ult_30_dias = ven_cases[-30:]
uru_cases_ult_30_dias = uru_cases[-30:]
col_cases_ult_30_dias = col_cases[-30:]
ecu_cases_ult_30_dias = ecu_cases[-30:]


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates_america, argentina_cases_ult_30_dias,'b',lw = 5)
plt.plot(adjusted_dates_america, bra_cases_ult_30_dias,'y',lw=5)
plt.plot(adjusted_dates_america,bol_cases_ult_30_dias,'g',lw =5)
plt.plot(adjusted_dates_america,chi_cases_ult_30_dias,'r',lw =5)
plt.plot(adjusted_dates_america,par_cases_ult_30_dias,'c',lw =5)
plt.plot(adjusted_dates_america,peru_cases_ult_30_dias,'m',lw =5)
plt.plot(adjusted_dates_america,ven_cases_ult_30_dias,'k',lw =5)
plt.plot(adjusted_dates_america,uru_cases_ult_30_dias,lw =5)
plt.plot(adjusted_dates_america,col_cases_ult_30_dias,lw =5)
plt.plot(adjusted_dates_america,ecu_cases_ult_30_dias,lw =5)


plt.title('Casos Confirmados COVID-19 - LATAM - Ultimos 30 dias', size=30)
plt.xlabel('Ultimos 30 dias', size=30)
plt.ylabel('# of Casos', size=30)
plt.legend(['Arg', 'Bra', 'Bol','Chi','Par','Peru','Ven','Uru','Col','Ecu'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
#argentina_recovered_ult_30_dias = argentina_recovered[-30:]
argentina_cases_ult_15_dias = argentina_cases[-15:]
bra_cases_ult_15_dias = bra_cases[-15:]
bol_cases_ult_15_dias = bol_cases[-15:]
chi_cases_ult_15_dias = chi_cases[-15:]
par_cases_ult_15_dias = par_cases[-15:]
peru_cases_ult_15_dias = peru_cases[-15:]
ven_cases_ult_15_dias = ven_cases[-15:]
uru_cases_ult_15_dias = uru_cases[-15:]
col_cases_ult_15_dias = col_cases[-15:]
ecu_cases_ult_15_dias = ecu_cases[-15:]


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates_america_test_15_dias, argentina_cases_ult_15_dias,'b',lw = 5)
plt.plot(adjusted_dates_america_test_15_dias, bra_cases_ult_15_dias,'y',lw=5)
plt.plot(adjusted_dates_america_test_15_dias,bol_cases_ult_15_dias,'g',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,chi_cases_ult_15_dias,'r',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,par_cases_ult_15_dias,'c',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,peru_cases_ult_15_dias,'m',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,ven_cases_ult_15_dias,'k',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,uru_cases_ult_15_dias,lw =5)
plt.plot(adjusted_dates_america_test_15_dias,col_cases_ult_15_dias,lw =5)
plt.plot(adjusted_dates_america_test_15_dias,ecu_cases_ult_15_dias,lw =5)


plt.title('Casos Confirmados COVID-19 - LATAM - Ultimos 15 dias', size=30)
plt.xlabel('Ultimos 15 dias', size=30)
plt.ylabel('# of Casos', size=30)
plt.legend(['Arg', 'Bra', 'Bol','Chi','Par','Peru','Ven','Uru','Col','Ecu'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
argentina_cases_ult_15_dias = argentina_cases[-15:]
bra_cases_ult_15_dias = bra_cases[-15:]
#bol_cases_ult_15_dias = bol_cases[-15:]
chi_cases_ult_15_dias = chi_cases[-15:]
#par_cases_ult_15_dias = par_cases[-15:]
#peru_cases_ult_15_dias = peru_cases[-15:]
#ven_cases_ult_15_dias = ven_cases[-15:]
#uru_cases_ult_15_dias = uru_cases[-15:]
#col_cases_ult_15_dias = col_cases[-15:]
ecu_cases_ult_15_dias = ecu_cases[-15:]


plt.figure(figsize=(16, 9))
plt.plot(adjusted_dates_america_test_15_dias, argentina_cases_ult_15_dias,'b',lw = 5)
plt.plot(adjusted_dates_america_test_15_dias, bra_cases_ult_15_dias,'y',lw=5)
#plt.plot(adjusted_dates_america_test,bol_cases_ult_15_dias,'g',lw =5)
plt.plot(adjusted_dates_america_test_15_dias,chi_cases_ult_15_dias,'r',lw =5)
#plt.plot(adjusted_dates_america_test_15_dias,par_cases_ult_15_dias,'c',lw =5)
#plt.plot(adjusted_dates_america_test_15_dias,peru_cases_ult_15_dias,'m',lw =5)
#plt.plot(adjusted_dates_america_test_15_dias,ven_cases_ult_15_dias,'k',lw =5)
#plt.plot(adjusted_dates_america_test_15_dias,uru_cases_ult_15_dias,lw =5)
#plt.plot(adjusted_dates_america_test_15_dias,col_cases_ult_15_dias,lw =5)
plt.plot(adjusted_dates_america_test_15_dias,ecu_cases_ult_15_dias,'g',lw =5)


plt.title('Casos Confirmados COVID-19 - LATAM - Top 3 + Argentina- Ultimos 15 dias', size=30)
plt.xlabel('Ultimos 15 dias', size=30)
plt.ylabel('# of Casos', size=30)
plt.legend(['Arg', 'Bra','Chi','Ecu'], prop={'size': 20})
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
world_mortality_rate_perc = mortality_rate[-1] * 100
world_mortality_rate_perc_round = np.round(world_mortality_rate_perc,4)


arg_mortality_rate_perc = arg_mortality_rate[-1] * 100
arg_mortality_rate_perc_round =np.round(arg_mortality_rate_perc,4)
arg_recovery_rate_perc = arg_recovery_rate[-1] * 100
arg_recovery_rate_perc_round = np.round(arg_recovery_rate_perc,4)

bra_mortality_rate_perc = bra_mortality_rate[-1] * 100
bra_mortality_rate_perc_round =np.round(bra_mortality_rate_perc,4)
bra_recovery_rate_perc = bra_recovery_rate[-1] * 100
bra_recovery_rate_perc_round = np.round(bra_recovery_rate_perc,4)

chi_mortality_rate_perc = chi_mortality_rate[-1] * 100
chi_mortality_rate_perc_round =np.round(chi_mortality_rate_perc,4)
chi_recovery_rate_perc = chi_recovery_rate[-1] * 100
chi_recovery_rate_perc_round = np.round(chi_recovery_rate_perc,4)

bol_mortality_rate_perc = bol_mortality_rate[-1] * 100
bol_mortality_rate_perc_round =np.round(bol_mortality_rate_perc,4)
bol_recovery_rate_perc = bol_recovery_rate[-1] * 100
bol_recovery_rate_perc_round = np.round(bol_recovery_rate_perc,4)

peru_mortality_rate_perc = peru_mortality_rate[-1] * 100
peru_mortality_rate_perc_round =np.round(peru_mortality_rate_perc,4)
peru_recovery_rate_perc = peru_recovery_rate[-1] * 100
peru_recovery_rate_perc_round = np.round(peru_recovery_rate_perc,4)

par_mortality_rate_perc = par_mortality_rate[-1] * 100
par_mortality_rate_perc_round =np.round(par_mortality_rate_perc,4)
par_recovery_rate_perc = par_recovery_rate[-1] * 100
par_recovery_rate_perc_round = np.round(par_recovery_rate_perc,4)

ven_mortality_rate_perc = ven_mortality_rate[-1] * 100
ven_mortality_rate_perc_round =np.round(ven_mortality_rate_perc,4)
ven_recovery_rate_perc = ven_recovery_rate[-1] * 100
ven_recovery_rate_perc_round = np.round(ven_recovery_rate_perc,4)

uru_mortality_rate_perc = uru_mortality_rate[-1] * 100
uru_mortality_rate_perc_round =np.round(uru_mortality_rate_perc,4)
uru_recovery_rate_perc = uru_recovery_rate[-1] * 100
uru_recovery_rate_perc_round = np.round(uru_recovery_rate_perc,4)

col_mortality_rate_perc = col_mortality_rate[-1] * 100
col_mortality_rate_perc_round =np.round(col_mortality_rate_perc,4)
col_recovery_rate_perc = col_recovery_rate[-1] * 100
col_recovery_rate_perc_round = np.round(col_recovery_rate_perc,4)



ecu_mortality_rate_perc = ecu_mortality_rate[-1] * 100
ecu_mortality_rate_perc_round =np.round(ecu_mortality_rate_perc,4)
ecu_recovery_rate_perc = ecu_recovery_rate[-1] * 100
ecu_recovery_rate_perc_round = np.round(ecu_recovery_rate_perc,4)

country_df_latam = pd.DataFrame({'Country Name': ['Argentina','Brasil','Chile','Bolivia','Paraguay','Peru','Venezuela','Uruguay','Colombia','Ecuador']
                                                 , 'Number of Confirmed Cases': [arg_confirmed_sum,bra_confirmed_sum,chi_confirmed_sum,bol_confirmed_sum,par_confirmed_sum,peru_confirmed_sum,ven_confirmed_sum,uru_confirmed_sum,col_confirmed_sum,ecu_confirmed_sum]
                                                 , 'Number of Deaths': [arg_death_sum,bra_death_sum,chi_death_sum,bol_death_sum,par_death_sum,peru_death_sum,ven_death_sum,uru_death_sum,col_death_sum,ecu_death_sum]
                                                 , 'Number of Recoveries': [recov_arg_sum,recov_bra_sum,recov_chi_sum,recov_bol_sum,recov_par_sum,recov_peru_sum,recov_ven_sum,recov_uru_sum,recov_col_sum,recov_ecu_sum]
                                                 , 'Mortality Rate':[arg_mortality_rate_perc_round,bra_mortality_rate_perc_round,chi_recovery_rate_perc_round,bol_mortality_rate_perc_round,par_mortality_rate_perc_round,peru_mortality_rate_perc_round,ven_mortality_rate_perc_round,uru_mortality_rate_perc_round,col_mortality_rate_perc_round,ecu_mortality_rate_perc_round] 
                                                 , 'Recovery Rate':[arg_recovery_rate_perc_round,bra_recovery_rate_perc_round,chi_recovery_rate_perc_round,bol_recovery_rate_perc_round,par_recovery_rate_perc_round,peru_recovery_rate_perc_round,ven_recovery_rate_perc_round,uru_recovery_rate_perc_round,col_recovery_rate_perc_round,ecu_recovery_rate_perc_round]})

country_latam_orden_casos = country_df_latam.sort_values('Number of Confirmed Cases',ascending=False)


country_latam_orden_casos.style.background_gradient(cmap='Greens')


#Ordenamos en base a Fallecimientos

country_latam_orden_casos = country_df_latam.sort_values('Number of Deaths',ascending=False)

country_latam_orden_casos.style.background_gradient(cmap='RdPu')
                                                                  


