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
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
import time
import itertools
import warnings
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
df = pd.read_csv('/kaggle/input/datos-ibex35-a-072020/DatosIbex2020.csv', encoding='utf-8')
df.head()
df['Fecha']=pd.to_datetime(df['Fecha'],format='%d.%m.%Y')
df.head()
df.tail()
ibex=df[['Fecha','Último']]
ibex.columns=['Fecha','Ibex']
ibex.info()
ibex['Ibex']=ibex['Ibex'].str.replace('.','')
ibex['Ibex']=ibex['Ibex'].str.replace(',','.')
ibex['Ibex']=ibex['Ibex'].astype('float')
ibex.head()
ibex=ibex[ibex['Fecha'] > '2020-04-01']
ibex.sort_values(by='Fecha',ascending=True,inplace=True)
ibex.head()
ibex.set_index('Fecha',inplace=True)
ibex['Ibex'].plot()
ibex['Ibex_log']=np.log(ibex['Ibex'])
ibex['Ibex_log'].plot()
ibex['Ibex_log_diff']=ibex['Ibex_log'].diff()
ibex['Ibex_log_diff'].plot()
y = ibex["Ibex"]
ibex['Ibex_mean']=ibex['Ibex'].mean()
y_media=ibex['Ibex_mean']

y_log=ibex["Ibex_log"]
ibex['Ibex_log_mean']=ibex['Ibex_log'].mean()
y_log_media=ibex['Ibex_log_mean']

y_log_diff=ibex["Ibex_log_diff"]
ibex['Ibex_log_diff_mean']=ibex['Ibex_log_diff'].mean()
y_log_diff_media=ibex['Ibex_log_diff_mean']

x=ibex.index
# visualización de los datos anteriores a los largo de los años
fig = plt.figure(figsize = (10, 10))
ax1, ax2, ax3 = fig.subplots(3, 1)

ax1.plot(x, y, label = "Serie Original")
ax1.plot(x, y_media, label = "Media de la Serie Original")
ax1.set_ylim(0, np.max(y)*1.3)
ax1.legend(loc = "upper left")

ax2.plot(x, y_log, label = "Serie Log.")
ax2.plot(x, y_log_media, label = "Media de la Serie Log.")
ax2.set_ylim(0, np.max(y_log)*1.3)
ax2.legend(loc = "lower left")

ax3.plot(x, y_log_diff, label = "Serie Logarítmica diferenciada")
ax3.plot(x, y_log_diff_media, label = "Media de la Serie. Log. Diff")
ax3.set_ylim(np.min(y_log_diff)*1.5, np.max(y_log_diff)*1.3)
ax3.legend(loc = "lower left")

fig.suptitle("Capturación de Pieles de Lince y sus transformaciones a lo largo de los años a lo largo de los años");
ibex['Ibex_log_diff'].isnull().sum()
ibex.dropna(axis=0, inplace=True)
LAGS = 24

fig = plt.figure(figsize = (10, 10))

((ax1, ax2), (ax3, ax4), (ax5, ax6)) = fig.subplots(3, 2)

# ----------------------------------------------------------------------------------------------------
# plot the data using the built in plots from the stats module
plot_acf(y, ax = ax1, lags = LAGS, title = "Autocorrelación")
plot_pacf(y, ax = ax2, lags = LAGS, title = "Autocorrelación Parcial")

plot_acf(y_log, ax = ax3, lags = LAGS, title = "Autocorrelación")
plot_pacf(y_log, ax = ax4, lags = LAGS, title = "Autocorrelación Parcial")

plot_acf(y_log_diff, ax = ax5, lags = LAGS, title = "Autocorrelación")
plot_pacf(y_log_diff, ax = ax6, lags = LAGS, title = "Autocorrelación Parcial")

fig.tight_layout()
serie_a_predecir = y
y_index = serie_a_predecir.index

date_train = int(len(y_index)*0.9)

y_train = serie_a_predecir[y_index[:date_train]]
y_test = serie_a_predecir[y_index[date_train:len(y_index)]]
y_train.tail()
y_test.head()
p = d = q = range(0, 6)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0, 0, 0, 0)]
st = time.time()

best_score = 0
best_params = None
best_seasonal_params = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            
            mod = sm.tsa.statespace.SARIMAX(y_train,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity = False,
                                            enforce_invertibility = False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            
            # guardamos el mejor resultado
            if best_score == 0:
                best_score = results.aic
                best_params = param
                best_seasonal_params = param_seasonal
                
            elif abs(results.aic) < abs(best_score):
                best_score = results.aic
                best_params = param
                best_seasonal_params = param_seasonal
            
        # alguna combinación de parámetros en SARIMAX, no son válidos
        # y los vamos a cazar con un except
        except:
            continue

et = time.time()
print("La búsqueda de parámetros no ha llevado {} minutos!".format((et - st)/60))

print("El mejor modelo es {}, \nCon un AIC de {}".format(best_params, best_score))
mod = sm.tsa.statespace.SARIMAX(y_train,
                                order = best_params,
                                seasonal_order = param_seasonal,
                                enforce_stationarity = False,
                                enforce_invertibility = False)

results = mod.fit()
results = mod.fit()

print(results.summary().tables[1])
# Para hacer una predicción es suficiente con especificar el número de steps/pasos futuros a estimar.
pred_uc = results.get_forecast(steps = len(y_test))

# Calcula el intervalo de confianza de la predicción.
pred_ci = pred_uc.conf_int()
pred_ci
predicted_values = pred_uc.predicted_mean.values
predicted_values
ax = serie_a_predecir.plot(label = 'Valores reales', figsize = (10, 10))
ax.plot(y_test.index, predicted_values, label = "Predicción")

ax.fill_between(y_test.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color = 'k', alpha = .25)

ax.set_xlabel('Dia')
ax.set_ylabel('Ibex 35')

plt.legend()
plt.show()
