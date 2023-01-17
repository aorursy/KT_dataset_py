import numpy as np

import pandas as pd



from sklearn import metrics

from math import sqrt



import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



import time

import itertools

import warnings

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.arima_model import ARIMA



warnings.filterwarnings("ignore")
# generamos una lista de 110 años para nuestre serie temporal 

años = np.arange(1821, 1931)



# la cantidad de pieles de linces captadas cada año



Pieles = [

269, 321, 585, 871, 1475, 2821, 3928, 5943, 4950, 2577,

523, 98, 184, 279, 409, 2285, 2685, 3409, 1824, 409,

151, 45, 68, 213, 546, 1033, 2129, 2536, 957, 361,

377, 225, 360, 731, 1638, 2725, 2871, 2119, 684, 299,

236, 245, 552, 1623, 3311, 6721, 4254, 687, 255, 473,

358, 784, 1594, 1676, 2251, 1426, 756, 299, 201, 229,

469, 736, 2042, 2811, 4431, 2511, 389, 73, 39, 49,

59, 188, 377, 1292, 4031, 3495, 587, 105, 387, 758,

1307, 3465, 6991, 6313, 3794, 1836, 345, 382, 808, 1388,

2713, 3800, 3091, 2985, 790, 674, 81, 80, 108, 229,

1132, 2432, 3574, 2935, 1537, 529, 485, 662, 1000, 1590

]
# generamos el dataframe completo de nuestro análisis

df = pd.DataFrame([años, Pieles]).T

df.columns = ["Año", "Pieles"]

df["Pieles_log"] = df["Pieles"].apply(np.log)

df["Pieles_log_diff"] = df["Pieles_log"].diff()

df["Año"] = pd.to_datetime(df["Año"], format = "%Y")

df.set_index("Año", inplace = True)

df.dropna(inplace = True, axis = "rows")

df.head()
# separar x y la y para el gráfico

x = df.index



y = df["Pieles"]

y_media = [np.mean(y) for _ in y]



y_log = df["Pieles_log"]

y_log_media = [np.mean(y_log) for _ in y_log]



y_log_diff = df["Pieles_log_diff"]

y_log_diff_mean = [np.mean(y_log_diff) for _ in y_log_diff]



# visualización de los datos anteriores a los largo de los años

fig = plt.figure(figsize = (10, 10))

ax1, ax2, ax3 = fig.subplots(3, 1)



# la serie original parece ser no estacionaria

# si nos fijamos en su comportamiento, vemos muchos picos y que la media de diferentes

# tramos de la serie es diferente.

# además la covarianza entre diferentes tramos también parece distinta.

ax1.plot(x, y, label = "Serie Original")

ax1.plot(x, y_media, label = "Media de la Serie Original")

ax1.set_ylim(0, np.max(y)*1.3)

ax1.legend(loc = "upper left")



# Si transformamos la serie utilizando el logaritmo neperiano (ln)

# tenemos una serie que YA es estacionaria en media y que oscila entorno

# a 7.

ax2.plot(x, y_log, label = "Serie Log.")

ax2.plot(x, y_log_media, label = "Media de la Serie Log.")

ax2.set_ylim(0, np.max(y_log)*1.3)

ax2.legend(loc = "lower left")



# Si aplicamos una diferenciación a al serie logarítmica, seguimos teniendo

# una serie estacionaria, pero esta vez, la media de la serie oscila entorno al cero.



# La diferenciación de una serie estacionaria SIEMPRE da lugar a otra serie estacionaria.

# Por este motivo, no haría falta hacer la diferencia y con la serie transformada (logarítmica)

# es suficiente.



ax3.plot(x, y_log_diff, label = "Serie Logarítmica diferenciada")

ax3.plot(x, y_log_diff_mean, label = "Media de la Serie. Log. Diff")

ax3.set_ylim(np.min(y_log_diff)*1.5, np.max(y_log_diff)*1.3)

ax3.legend(loc = "lower left")



fig.suptitle("Capturación de Pieles de Lince y sus transformaciones a lo largo de los años a lo largo de los años");


for serie, nombre_serie in zip([y, y_log, y_log_diff], ["Serie Original", "Serie Log.", "Serie. Log. Diff"]):

    

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
serie_a_predecir = y_log
y_index = serie_a_predecir.index



date_train = int(len(y_index)*0.9)



y_train = serie_a_predecir[y_index[:date_train]]

y_test = serie_a_predecir[y_index[date_train:len(y_index)]]
y_train.tail()
y_test.head()
# Para hacer el gridsearch, vamos a calcular los posibles valores que vamos a pasarle al modelo.

p = d = q = range(0, 3)

pdq = list(itertools.product(p, d, q))



# Vamos a utilizar el modelo SARIMAX, porque en su implementaciòn en Python existen herramientas adicionales

# que nos facilitan el análisis y que no están disponibles en la implementación de ARIMA.



# Ahora bien, SARIMAX es un modelo ARIMA pero con un componente estacional (la leta S de Seasonal) y también

# un componente exógeno (X de eXogenous regressors)

# Por tanto un modelo SARIMAX de (1, 1, 1) x (0, 0, 0, 0)

# Es un modelo ARIMA (1, 1, 1)



# En caso de querer probar un modelo SARIMAX completo, ejecutar la siguiente línea de itertools.

# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



# Los dejamos a cero para sólo analizar un modelo ARIMA.

seasonal_pdq = [(0, 0, 0, 0)]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[0]))

print('SARIMAX: {} x {}'.format(pdq[3], seasonal_pdq[0]))
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
# Entrenamos el modelo con los mejores parametros.



mod = sm.tsa.statespace.SARIMAX(y_train,

                                order = best_params,

                                seasonal_order = param_seasonal,

                                enforce_stationarity = False,

                                enforce_invertibility = False)



results = mod.fit()
results = mod.fit()



print(results.summary().tables[1])
results.plot_diagnostics(figsize = (15, 12), lags = 3);
# Para hacer una predicción es suficiente con especificar el número de steps/pasos futuros a estimar.

pred_uc = results.get_forecast(steps = len(y_test))



# Calcula el intervalo de confianza de la predicción.

pred_ci = pred_uc.conf_int()
ax = serie_a_predecir.plot(label = 'Valores reales', figsize = (20, 15))



pred_uc.predicted_mean.plot(ax = ax, label = 'Predicción')



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color = 'k', alpha = .25)



ax.set_xlabel('Año')

ax.set_ylabel('Pieles Capturadas')



plt.legend()

plt.show()
y_pred = pred_ci.iloc[:, 0]
# El RMSE es de 2.52

rmse = sqrt(metrics.mean_squared_error(y_test, y_pred))



print("El modelo ARIMA con los parametros {}, ha dado un rmse en test de {}".format(best_params, round(rmse, 2)))