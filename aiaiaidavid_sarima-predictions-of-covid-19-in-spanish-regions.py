import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import mean_squared_error
from math import sqrt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = True

# Set precision to two decimals
pd.set_option("display.precision", 2)
df = pd.read_csv('../input/covid19-cases-in-spain-by-ccaa-26042020/serie_historica_acumulados_dv_26042020.csv', sep=',', encoding = 'unicode_escape')
# Convert FECHA column to datetime format
df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y')
# Add mortality column
# Mortality = Nro Fallecidos / Nro Casos (por fecha)
df['MORTALITY'] = df['FALLECIDOS']/df['CASOS']
df.head()
# Replace NaN with 0
df.fillna(value=0, inplace=True)
# Get list of CCAA
codes_CCAA = df['CCAA'].unique()
print(len(codes_CCAA), codes_CCAA)
# Full CA name and CA coede pairs
dict_CCAA = {'AN': 'Andalucia', 'AR': 'Aragon', 'AS': 'Asturias', 
             'IB': 'Islas Baleares', 'CN': 'Islas Canarias', 
             'CB': 'Cantabria', 'CM': 'Castilla La Mancha', 
             'CL': 'Castilla y Leon', 'CT': 'Catalunya', 'CE': 'Ceuta', 
             'VC': 'Comunidad Valenciana', 'EX': 'Extremadura',
             'GA': 'Galicia', 'MD': 'Comunidad de Madrid', 'ME': 'Melilla',
             'MC': 'Murcia', 'NC': 'Comunidad Navarra' , 'PV': 'Pais Vasco',
             'RI': 'La Rioja'}
# Create a dictionary of dataframes (one for each CCAA)
array_df_CCAA = {}
for x in codes_CCAA:
  array_df_CCAA[x] = pd.DataFrame(df[df['CCAA']==x])
###########################################
# Choose one CA
###########################################
chosen_CA = 'VC'
dfp = array_df_CCAA[chosen_CA]
dfp.set_index('FECHA', inplace=True)
print("Your choice of CCAA: " + dict_CCAA[chosen_CA])
# Create new columns with daily difference (casos(t)-casos(t-1))
dfp['CASOS_DIFF'] = dfp['CASOS'].diff()
dfp['HOSPITAL_DIFF'] = dfp['HOSPITAL'].diff()
dfp['UCI_DIFF'] = dfp['UCI'].diff()
dfp['FALLECIDOS_DIFF'] = dfp['FALLECIDOS'].diff()
dfp['RECUPERADOS_DIFF'] = dfp['RECUPERADOS'].diff()
# Fill first element of the diff series with 0
dfp.fillna(value=0, inplace=True )
# Make list of cases and daily change rates
casos = ('CASOS', 'HOSPITAL', 'UCI', 'FALLECIDOS', 'RECUPERADOS') 
casos_daily_diff = ('CASOS_DIFF', 'HOSPITAL_DIFF', 'UCI_DIFF', 'FALLECIDOS_DIFF', 'RECUPERADOS_DIFF')
# Plot cases
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('NUMERO DE CASOS')
plt.title('Evolucion temporal de los casos de Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')

plt.plot(dfp.index, dfp.CASOS, linewidth=1, marker='')
plt.plot(dfp.index, dfp.HOSPITAL, linewidth=1, marker='')
plt.plot(dfp.index, dfp.UCI, linewidth=1, marker='')
plt.plot(dfp.index, dfp.FALLECIDOS, linewidth=1, marker='')
plt.plot(dfp.index, dfp.RECUPERADOS, linewidth=1, marker='')

plt.legend(casos, loc='upper left', fontsize='large')
plt.show()
# Plot daily change (value(t)-value(t-1))
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('VARIACION DIARIA')
plt.title('Variacion diaria de los casos de Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')

plt.plot(dfp.index, dfp.CASOS_DIFF, linewidth=1, marker='')
plt.plot(dfp.index, dfp.HOSPITAL_DIFF, linewidth=1, marker='')
plt.plot(dfp.index, dfp.UCI_DIFF, linewidth=1, marker='')
plt.plot(dfp.index, dfp.FALLECIDOS_DIFF, linewidth=1, marker='')
plt.plot(dfp.index, dfp.RECUPERADOS_DIFF, linewidth=1, marker='')

plt.legend(casos_daily_diff, loc='upper left', fontsize='large')
plt.show()
# Split data into train and test subsets 
nbr_predictions = 5
dfp_train = dfp[:len(dfp)-nbr_predictions]
dfp_test = dfp[len(dfp)-nbr_predictions:]
dfp_test
plt.figure(figsize=(9,4))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos confirmados de Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp.index, dfp.CASOS, linewidth=1)
# Check for seasonality and trend information
from statsmodels.tsa.seasonal import seasonal_decompose
s = seasonal_decompose(dfp_train.CASOS, model = "add")
s.plot();
# Execute auto_arima to find optimal parameters
!pip install pmdarima
from pmdarima import auto_arima      
auto_arima(dfp_train.CASOS, seasonal=True, m=7,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4, suppress_warnings=True).summary()
# Configure model and fit it with train data
import statsmodels.api as sm
ARIMA_model = sm.tsa.statespace.SARIMAX(dfp_train.CASOS, order=(0,2,1), seasonal_order=(0,0,0,0), suppress_warnings=True).fit(maxiter=500)
# Create y_predicted as a copy of dfp_test so we keep all data in one dataframe
y_predicted = dfp_test.copy()
y_predicted['SARIMA_CASOS'] = ARIMA_model.forecast(len(dfp_test.CASOS))
y_predicted
# Plot of case predictions: CONFIRMED CASES
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('CASOS confirmados de Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp_train.index, dfp_train.CASOS, linewidth=1, marker='x', label='Training')
plt.plot(dfp_test.index, dfp_test.CASOS, linewidth=1,  marker='x', label='Test')
plt.plot(dfp_test.index, y_predicted['SARIMA_CASOS'], linewidth=1, marker='o', markersize=4, label='SARIMAX predicted')
plt.legend(loc='upper left', fontsize='large')
plt.show()
# Check accuracy of model with Root Mean Square Error
import statsmodels.api as sm
rms = sqrt(mean_squared_error(dfp_test.CASOS, y_predicted.SARIMA_CASOS))
print('RMSE ARIMA = ' + str(rms))
plt.figure(figsize=(9,4))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos en HOSPITAL de Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp.index, dfp.HOSPITAL, linewidth=1)
# Check for seasonality and trend information
s = seasonal_decompose(dfp_train.HOSPITAL, model = "add")
s.plot();
# Execute auto_arima to find optimal parameters
auto_arima(dfp_train.HOSPITAL, seasonal=True, m=7,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4, suppress_warnings=True).summary()
# Configure model and fit it with train data
ARIMA_model = sm.tsa.statespace.SARIMAX(dfp_train.HOSPITAL, order=(0,2,0), seasonal_order=(0,0,0,0), suppress_warnings=True).fit(maxiter=500)
# Add a new column to y_predicted with the HOSPITAL cases prediction
y_predicted['SARIMA_HOSPITAL'] = ARIMA_model.forecast(len(dfp_test.HOSPITAL)) 
y_predicted
# Plot of case predictions: HOSPITAL CASES
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos en HOSPITAL con Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp_train.index, dfp_train.HOSPITAL, linewidth=1, marker='x', label='Training')
plt.plot(dfp_test.index, dfp_test.HOSPITAL, linewidth=1,  marker='x', label='Test')
plt.plot(dfp_test.index, y_predicted['SARIMA_HOSPITAL'], linewidth=1, marker='o', markersize=4, label='SARIMAX predicted')
plt.legend(loc='upper left', fontsize='large')
plt.show()
# Check accuracy of model with Root Mean Square Error
rms = sqrt(mean_squared_error(dfp_test.HOSPITAL, y_predicted.SARIMA_HOSPITAL))
print('RMSE ARIMA = ' + str(rms))
plt.figure(figsize=(9,4))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos en la UCI con Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp.index, dfp.UCI, linewidth=1)
# Check for seasonality and trend information
s = seasonal_decompose(dfp_train.UCI, model = "add")
s.plot();
# Execute auto_arima to find optimal parameters
auto_arima(dfp_train.UCI, seasonal=True, m=7,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4, suppress_warnings=True).summary()
# Configure model and fit it with train data
ARIMA_model = sm.tsa.statespace.SARIMAX(dfp_train.UCI, order=(1,2,2), seasonal_order=(1,0,0,7), suppress_warnings=True).fit(maxiter=500)
# Add a new column to y_predicted with the UCI cases prediction
y_predicted['SARIMA_UCI'] = ARIMA_model.forecast(len(dfp_test.UCI)) 
y_predicted
# Plot of case predictions: UCI CASES
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos en la UCI con Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')

plt.plot(dfp_train.index, dfp_train.UCI, linewidth=1, marker='x', label='Training')
plt.plot(dfp_test.index, dfp_test.UCI, linewidth=1,  marker='x', label='Test')
plt.plot(dfp_test.index, y_predicted['SARIMA_UCI'], linewidth=1, marker='o', markersize=4, label='SARIMAX predicted')

plt.legend(loc='upper left', fontsize='large')
plt.show()
# Check accuracy of model with Root Mean Square Error
rms = sqrt(mean_squared_error(dfp_test.UCI, y_predicted.SARIMA_UCI))
print('RMSE ARIMA = ' + str(rms))
plt.figure(figsize=(9,4))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos FALLECIDOS con Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')
plt.plot(dfp.index, dfp.FALLECIDOS, linewidth=1)
# Check for seasonality and trend information
s = seasonal_decompose(dfp_train.FALLECIDOS, model = "add")
s.plot();
# Execute auto_arima to find optimal parameters
auto_arima(dfp_train.FALLECIDOS, seasonal=True, m=7,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4, suppress_warnings=True).summary()
# Configure model and fit it with train data
ARIMA_model = sm.tsa.statespace.SARIMAX(dfp_train.FALLECIDOS, order=(0,2,1), seasonal_order=(0,0,0,0)).fit(maxiter=500)
# Add a new column to y_predicted with the FALLECIDOS cases prediction (deaths)
y_predicted['SARIMA_FALLECIDOS'] = ARIMA_model.forecast(len(dfp_test.FALLECIDOS)) 
y_predicted
# Plot of case predictions: FALLECIDOS CASES
plt.figure(figsize=(12,6))
plt.xlabel ('FECHA')
plt.ylabel('CASOS')
plt.title('Casos FALLECIDOS con Covid-19:  ' + dict_CCAA[chosen_CA], fontsize='x-large')

plt.plot(dfp_train.index, dfp_train.FALLECIDOS, linewidth=1, marker='x', label='Training')
plt.plot(dfp_test.index, dfp_test.FALLECIDOS, linewidth=1,  marker='x', label='Test')
plt.plot(dfp_test.index, y_predicted['SARIMA_FALLECIDOS'], linewidth=1, marker='o', markersize=4, label='SARIMAX predicted')

plt.legend(loc='upper left', fontsize='large')
plt.show()
# Check accuracy of model with Root Mean Square Error
rms = sqrt(mean_squared_error(dfp_test.FALLECIDOS, y_predicted.SARIMA_FALLECIDOS))
print('RMSE ARIMA = ' + str(rms))