import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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
df_cur = pd.read_csv("/kaggle/input/euro-exchange-daily-rates-19992020/euro-daily-hist_1999_2020.csv", parse_dates=["Period\\Unit:"])
df_cur.sample(5)
df_cur.info(verbose=True)
# Solo me interesa la del USD Dolar contar el Euro
df_cur[ ['Period\\Unit:' , '[US dollar ]'] ] 
df=df_cur[ ['Period\\Unit:' , '[US dollar ]'] ] 
df.columns=['Fecha','USD']
df.head()
df.info()
# Tiene caracteres "Nulos"
df['USD'].describe()
df[df['USD']=='-']['Fecha'].min()
df[df['USD']=='-']['Fecha'].max()
# Me voy a quedar con los que no tienen nulos. Desde el 01/05/2012. Mas que suficiente para probar
df=df[ df['Fecha'] > '2012-05-01 00:00:00']
df['USD']=df['USD'].astype(np.number)
df.set_index('Fecha',inplace=True)

# Pintamos la serie iniicial:
df['USD'].plot()
# Nos vamos a coger desde 2019 a 2020, mas que suficiente:
df=df[ df.index > '2019-01-01 00:00:00']
# Pintamos la serie inicial
df['USD'].plot()
df['USD_log']=np.log(df['USD'])
df.head()
# Pintamos la serie logaritmica
df['USD_log'].plot()
# Transformamos la serie a diferencias logaritmicas
df['USD_log_diff']=df['USD_log'].diff()
# Pintamos la serie de diferencias
df['USD_log_diff'].plot()
df['USD_log_diff2']=df['USD_log_diff'].diff()
# Pintamos esta serie
df['USD_log_diff2'].plot()
df.dropna(axis=0,inplace=True)
x=df.index
y = df["USD"]
df['USD_mean']=df['USD'].mean()
y_media=df['USD_mean']
y_log=df["USD_log"]
df['USD_log_mean']=df['USD_log'].mean()
y_log_media=df['USD_log_mean']
y_log_diff=df["USD_log_diff"]
df['USD_log_diff_mean']=df['USD_log_diff'].mean()
y_log_diff_media=df['USD_log_diff_mean']
y_log_diff2=df["USD_log_diff2"]
df['USD_log_diff2_mean']=df['USD_log_diff2'].mean()
y_log_diff2_media=df['USD_log_diff2_mean']
# visualización de los datos anteriores a los largo de los años
fig = plt.figure(figsize = (10, 10))
ax1, ax2, ax3, ax4 = fig.subplots(4, 1)

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

ax4.plot(x, y_log_diff2, label = "Serie Logarítmica diferenciada dos veces")
ax4.plot(x, y_log_diff2_media, label = "Media de la Serie. Log. Diff")
ax4.set_ylim(np.min(y_log_diff2)*1.5, np.max(y_log_diff2)*1.3)
ax4.legend(loc = "lower left")

fig.suptitle("Capturación de Pieles de Lince y sus transformaciones a lo largo de los años a lo largo de los años");

for serie, nombre_serie in zip([y, y_log, y_log_diff, y_log_diff2], ["Serie Original", "Serie Log.", "Serie. Log. Diff", "Serie. Log. Diff2"]):
    
    print("------------------------------------------------------------------")
    
    print("Estamos trabajando con la serie {}\n".format(nombre_serie))
    resultado_analisis = adfuller(serie)
    
    valor_estadistico_adf = resultado_analisis[0]
    p_valor = resultado_analisis[1]
    
    print("Valor estadistico de ADF de las tablas precalculadas: {}".format(-2.89))
    print("Valor estadistico de ADF: {}\n".format(valor_estadistico_adf))
    
    print("Nivel de significación para tomar la serie como estacionaria {}".format(0.05))
    print("p-valor: {}\n".format(p_valor))
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
# Para la serie log_diff  vemos en la autocorrelacion parcial un AR(1)
# en la autocorrelacion evmos un MA de 1 tambien. Asi que seria un ARMA(1,1)
serie_a_predecir = y_log_diff
serie_a_predecir.head()
serie_a_predecir_df=pd.DataFrame(serie_a_predecir)
serie_a_predecir_df.sort_values(by='Fecha',inplace=True)
serie_a_predecir=serie_a_predecir_df['USD_log_diff']
serie_a_predecir
y_index = serie_a_predecir.index
date_train = int(len(y_index)*0.9)

y_train = serie_a_predecir[y_index[:date_train]]
y_test = serie_a_predecir[y_index[date_train:len(y_index)]]
y_train.index.min(),y_train.index.max()
y_test.index.min(),y_test.index.max()
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(0, 0, 0, 0)]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[0]))
print('SARIMAX: {} x {}'.format(pdq[3], seasonal_pdq[0]))
pdq
seasonal_pdq = [(0, 0, 0, 0)]
best_score = 0
best_params = None
salida={}
for param in pdq:    
    mod = sm.tsa.statespace.SARIMAX(y_train,
                                    order=param,
                                    seasonal_order=(0,0,0,0),
                                    enforce_stationarity = False,
                                    enforce_invertibility = False)

    results = mod.fit()
    print('ARIMA{}x{}12 - AIC:{}'.format(param, (0,0,0,0), results.aic))
    # Meto los resultados en un diccionario
    salida[param]=results.aic
salida
# Hago un DataFrame y ordeno para ver cual es el menor.
salida_df=pd.DataFrame(index=salida.keys(), data=salida.values())
salida_df.sort_values(by=0,ascending=False)
# Tiramos el modelo
mod = sm.tsa.statespace.SARIMAX(y_train,
                                order = (1,0,0),
                                seasonal_order = (0,0,0,0),
                                enforce_stationarity = False,
                                enforce_invertibility = False)

results = mod.fit()
results = mod.fit()

print(results.summary().tables[1])
# Para hacer una predicción es suficiente con especificar el número de steps/pasos futuros a estimar.
pred_uc = results.get_forecast(steps = len(y_test))

# Calcula el intervalo de confianza de la predicción.
pred_ci = pred_uc.conf_int()
len(pred_uc)
ax = serie_a_predecir.plot(label = 'Valores reales', figsize = (20, 15))

pred_uc.predicted_mean.plot(ax = ax, label = 'Predicción')

#ax.fill_between(pred_ci.index,
#                pred_ci.iloc[:, 0],
#                pred_ci.iloc[:, 1], color = 'k', alpha = .25)

ax.set_xlabel('Año')
ax.set_ylabel('Pieles Capturadas')

plt.legend()
plt.show()
