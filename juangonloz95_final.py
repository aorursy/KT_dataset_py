import os



# ESTABLECIENDO EL DIRECTORIO DE TRABAJO

CWD = "/kaggle/input/"

os.chdir(CWD)

os.getcwd()

os.listdir()
from math import * #mathematic package



import pandas as pd #panel data



from matplotlib import pyplot as plt #graphics 



import numpy as np #numeric python

from numpy import diff # diferenciation



#import yfinance as yf # yahoo api



from scipy.stats import norm  #normality test

from scipy.stats import jarque_bera #jarque-bera

from scipy.stats import shapiro # TEST SHAPIRO-WILK 

from scipy.stats import normaltest # D'AGOSTINO K^2 TEST

from scipy.stats import anderson # TEST DE ANDERSON DARLING





!pip install PyPR

#from pypr.stattest.ljungbox import * # TEST DE LJUNG-BOX



import scipy.stats 



from dateutil.parser import parse #time series



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ACF and PACF

from statsmodels.tsa.ar_model import AR #AR model

from statsmodels.tsa.arima_model import ARMA #ARMA model

from statsmodels.tsa.arima_model import ARIMA #ARIMA MODEL

from statsmodels.tsa.stattools import arma_order_select_ic #order of ARMA model 

from statsmodels.graphics.gofplots import qqplot #qqplot

from statsmodels.tsa.stattools import adfuller #test adfuller
# IMPORTAR LA BASE DE DATOS DE LOGITECH

LOGI = pd.read_csv('logi-1/LOGI.csv',  parse_dates=['Date'], index_col='Date')



type(LOGI.index)
LOGI.head()
# CREANDO FUNCION DE ESTADISTICAS BASICAS



def STATS(X):

    print("mean = ",X.mean())

    print("s.d. = ",X.std())

    print("var = ",X.var())

    print("vol = ",X.std()*sqrt(252))

    print("skew = ",X.skew())

    print("Kurt = ",X.kurt())

    print("t_s = ",sqrt(X.count())*X.skew()/sqrt(6))

    print("t_k = ",sqrt(X.count())*(X.kurt()-3)/sqrt(24))
result = adfuller(LOGI['Adj Close'])

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')



# INTERPRETACION Augmented Dickie Fuller



alpha = 0.05

if result[1] > alpha:

	print('H0: no estacionario, existe raiz unitaria (el p-value>0.05 no rechaza H0)')

else:

	print('Ha: es estacionaria, no existe raiz unitaria (el p-value<0.05; rechaza H0)')
# DIFERENCIANDO LA VARIABLE

LOGI_DIF = diff(LOGI['Adj Close'])

plt.plot(LOGI_DIF)

plt.show()



## SE PUEDE VER QUE DESPUES DE UNA DIFERENCIACIO TOMA

## APARIENCIA DE UNA CAMINATA ALEATORIA CON MEDIA 0
result = adfuller(LOGI_DIF)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')



# INTERPRETACION Augmented Dickie Fuller



alpha = 0.05

if result[1] > alpha:

	print('H0: no estacionario, existe raiz unitaria (el p-value>0.05 no rechaza H0)')

else:

	print('Ha: es estacionaria, no existe raiz unitaria (el p-value<0.05; rechaza H0)')
STATS(LOGI["Adj Close"])



## OBSERVACION SOBRE LA KURTOSIS: UNA DISTRIB NORMAL SE COMPORTA CON KURT=3

## ESTA TIENE UNA KURTOSIS DE -0.93
# Obtain the log-returns

d_ret = np.log(LOGI['Adj Close']).diff()

# Eliminate NaN

d_ret = d_ret[~np.isnan(d_ret)]

# Percentage returns

d_ret = d_ret*100
plt.plot(d_ret)

plt.show()
print("Statistics for log-returns:")

STATS(d_ret)
# VOLVIENDO A GRAFICAR CON UNA DIFERENCIACION

plot_acf(d_ret, lags=20)

plt.show()



plot_pacf(d_ret, lags=20)

plt.show()



## EN ESTE CASO SE VE QUE YA NO ESTA CORRELACIONADO

## CON LAS OBSERVACIONES DE PERIODOS POSTERIORES



## BASADO EN LOS RESULTADOS SE EVIDENCIA QUE TIENE UN 

## PROCESOS AUTORREGRESIVO DE ORDEN 1
## ESTE ULTIMO DA EL ORDEN QUE NECESITO PARA REPLICAR EL MODELO



# MODELO AJUSTADO

model = AR(d_ret)



# ENCONTRANDO EL ORDEN AUTORREGRESIVO DEL MODELO

ret_order = model.select_order(maxlag=20, ic= 'aic')

print("order: ",ret_order) 



# ASUMIENDO QUE NO HAY TENDENCIA, ENCUENTRA EL ORDEN DEL MODELO

print("Arma order AIC: ", arma_order_select_ic(d_ret, ic='aic', trend='nc').aic_min_order)



## NO CREEMOS QUE TENGA ESTE ORDEN EL MODELO ARMA PORQUE

## HABIAMOS IDENTIFICADO QUE TENIA ORDEN AR(1) Y UN PERIODO DE DIFERENCIACION
# ESTIMANDO LOS PARAMETROS DEL MODELO

model = ARMA(d_ret, order=(4,1))

model_fit = model.fit(disp=0)



# GENERANDO EL RESUMEN DEL MODELO

print(model_fit.summary())



## PREOCUPACION: LOS P-VALUES DE ESTA MUESTRA ESTAN FKING LOW
# AJUSTE DEL MODELO, SI IC NO ESTA ESPECIFICADO LO AJUSTARA A EL NUMERO MAXIMO DE RETRASOS

# LAGS, DE LO CONTRARIO ENCAJARA EL MEJOR AL MAXIMO NUMERO DE RETRASOS.

# Fit the model, if ic is not specified it will fit maxlag, else it will fit the best up to maxlag

model_fit = model.fit(maxlag=ret_order+1, ic='aic', trend='nc')



# model_fit is type ARResult type. Look for the documentarion online on: statsmodels.tsa.ar_model

# https://www.statsmodels.org/dev/_modules/statsmodels/tsa/ar_model.html#AR.fit

print(model_fit.summary())
print('Lag: %s' % model_fit.k_ar)

print('phi_i Coefficients: %s' % model_fit.params)

print("p_values: ", model_fit.pvalues)

# print("roots: ", model_fit.roots)

print("AIC: ", model_fit.aic)
# Calculate the phi_0 for the model

mean = sum(d_ret)/(len(d_ret)) 

mu = model_fit.params[0]

phi1 = model_fit.params[1]

phi2 = model_fit.params[2]

phi0 = (1.0-phi1-phi2)*mu

print(phi0)



# sigma2: The variance of the residuals ## Residual standard error

print(sqrt(model_fit.sigma2))
# COMPROBACION DE NORMALIDAD



## TEORICAMENTE, LOS RETORNOS DE UN ACTIVO LIQUIDO DEBEN COMPORTARSE

## NORMALES CON ERROR RUIDO BLANCO GAUSSIANO



# GRAFICAS UTILES PREVIOS TEST DE NORMALIDAD



# HISTOGRAMA

plt.hist(d_ret, normed = True)

plt.xlabel('Precios')

plt.ylabel('Probabilidad')

plt.title('Histograma de la distribucion')

plt.show()



# CUANTILES

qqplot(LOGI_DIF, line = 's')

plt.show()
stat, p = shapiro(d_ret)

print('Statistics=%.3f, p=%.3f' % (stat, p))



# INTERPRETACION

alpha = 0.05

if p > alpha:

	print('Sample looks Gaussian (fail to reject H0)')

else:

	print('Sample does not look Gaussian (reject H0)')
stat, p = normaltest(d_ret)

print('Statistics=%.3f, p=%.3f' % (stat, p))



# INTERPRETACION

alpha = 0.05

if p > alpha:

	print('Sample looks Gaussian (fail to reject H0)')

else:

	print('Sample does not look Gaussian (reject H0)')
result = anderson(d_ret)



# INTERPRETACION

print('Statistic: %.3f' % result.statistic)

p = 0

for i in range(len(result.critical_values)):

	sl, cv = result.significance_level[i], result.critical_values[i]

	if result.statistic < result.critical_values[i]:

		print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))

	else:

		print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
result = adfuller(d_ret)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')



# INTERPRETACION Augmented Dickie Fuller



alpha = 0.05

if result[1] > alpha:

	print('H0: no estacionarios, existe raiz unitaria (el p-value>0.05 no rechaza H0)')

else:

	print('Ha: es estacionaria, no existe raiz unitaria (el p-value<0.05; rechaza H0)')
from statsmodels.stats.diagnostic import acorr_ljungbox

resid_model_fit = model_fit.resid

print(acorr_ljungbox(resid_model_fit, lags = 10)) #ljung box
ret = pd.DataFrame(d_ret)
print( phi0, phi1, phi2)
reto = [0]*250

for i in range(0,250):

    reto[i] = d_ret[i]

print(reto)
print(d_ret)
# MODELO DE PREDICCION



predn = 30 # NUMERO DE MODELOS A PREDECIR



for i in range(predn):

    reto.append(phi0 + (phi1 * reto[i-1]) + (phi2 * reto[i-2]))
plt.plot([i for i in range(len(reto)-predn)], reto[:-predn], 'o-')

plt.plot([i for i in range(len(reto)-predn-1,len(reto))], reto[len(reto)-predn-1:], 'o-')

plt.show()



# Aditionally we could predict with the prediction function

model_fit.plot_predict(99, 115, dynamic=True, plot_insample=False)

plt.show()
SPX = pd.read_csv("sp500/SPX.csv",parse_dates=['Date'], index_col='Date')

spx = SPX['Adj Close']

ret_spx = np.log(spx).diff()

ret_spx = ret_spx[~np.isnan(ret_spx)]

ret_spx = ret_spx * 100
STATS(ret_spx)
plt.plot(ret_spx)

plt.plot()
plot_acf(ret_spx, lags=20)

plot_pacf(ret_spx, lags=20)

plt.show()
ret2_spx = ret_spx.pow(2)



plot_acf(ret2_spx, lags=20)

plt.show()



plot_pacf(ret2_spx, lags=20)

plt.show()
!pip install arch



from arch import arch_model



am = arch_model(ret_spx, p=4, o=0, q=2) 

am_spx_ag = am.fit()

print(am_spx_ag.summary())
am_spx_ag.plot()

plt.show()
ar_spx = ARIMA(ret_spx, order=(3,1,1))

ar_est_spx = ar_spx.fit(disp=0)

print(ar_est_spx.summary())
print(sqrt(ar_est_spx.sigma2))



mu = ar_est_spx.params[0]

phi1 = ar_est_spx.params[1]

phi2 = ar_est_spx.params[2]

phi3 = ar_est_spx.params[3]

phi0 = (1-phi1-phi2-phi3)*mu

print(phi0)
am = arch_model(ret_spx, mean='AR', lags=3, p=4, o=0, q=2)

am_adj = am.fit(disp = 0)

print(am_adj.summary())

mmu = am_adj.params["Const"]

phi1 = am_adj.params["Adj Close[1]"]

phi2 = am_adj.params["Adj Close[2]"]

phi3 = am_adj.params["Adj Close[3]"]

print("phi0 =", (1-phi1-phi2-phi3)*mmu)
am_adj.plot()

plt.show()
import seaborn

import matplotlib.mlab as mlab

from scipy.stats import norm

from tabulate import tabulate
var_spx_90 = norm.ppf(1-0.9 , ret_spx.mean() , ret_spx.std())

var_spx_95 = norm.ppf(1-0.95 , ret_spx.mean() , ret_spx.std())

var_spx_99 = norm.ppf(1-0.99 , ret_spx.mean() , ret_spx.std())
print(tabulate([["90%", var_spx_90],["95%", var_spx_95],["99%", var_spx_99]], headers=["confidencelevel","value at risk"]))
vares = [0]*100



for i in range(0,100):

    vares[i] = norm.ppf(0.01*i, ret_spx.mean() , ret_spx.std())

print(vares)



plt.plot(vares)

plt.show()
es_spx_90 = 0.1 * norm.pdf(var_spx_90)

es_spx_95 = 0.05 * norm.pdf(var_spx_95)

es_spx_99 = 0.01 * norm.pdf(var_spx_99)
print(tabulate([["90%", es_spx_90],["95%", es_spx_95],["99%", es_spx_99]], headers=["confidencelevel","expected shorfall"]))
from statsmodels.stats.stattools import durbin_watson

from statsmodels.stats.diagnostic import acorr_ljungbox
am_spx_ag_resid = am_spx_ag.resid

am_spx_ag_resid = am_spx_ag_resid[~np.isnan(am_spx_ag_resid)]

print(am_spx_ag_resid)

#

ar_est_spx_resid = ar_est_spx.resid

ar_est_spx_resid = ar_est_spx_resid[~np.isnan(ar_est_spx_resid)]

print(ar_est_spx_resid)

#

am_adj_resid = am_adj.resid

am_adj_resid = am_adj_resid[~np.isnan(am_adj_resid)]

print(am_adj_resid)
print('white noise')

print(tabulate([['Modelo de media constante GARCH: ', durbin_watson(am_spx_ag_resid)],

                ['Modelo ARIMA(3,1,1): ', durbin_watson(ar_est_spx_resid)],

                ['Modelo AR-ARCH(3;4,0,2): ', durbin_watson(am_adj_resid)]],

                headers=["Regresion","Estadistico"]))

#-----------------------------------------------------

print('normalidad de los residuos (jarque bera)')

print(tabulate([['Modelo de media constante GARCH: ', jarque_bera(am_spx_ag_resid)],

                ['Modelo ARIMA(3,1,1): ', jarque_bera(ar_est_spx_resid)],

                ['Modelo AR-ARCH(3;4,0,2): ', jarque_bera(am_adj_resid)]], 

                headers=["Regresion","Estadistico"]))

#-----------------------------------------------------

print('Test de Ljung-Box')

print(tabulate([['Modelo de media constante GARCH: ', acorr_ljungbox(am_spx_ag_resid, lags = 10)],

                ['Modelo ARIMA(3,1,1): ', acorr_ljungbox(ar_est_spx_resid, lags = 10)],

                ['Modelo AR-ARCH(3;4,0,2): ', acorr_ljungbox(am_adj_resid, lags = 10)]],

                headers=["Regresion","Estadistico"]))
print('Modelo de media constante GARCH')

plot_acf(am_spx_ag_resid, lags=20)

plt.show()



plot_pacf(am_spx_ag_resid, lags=20)

plt.show()
print('Modelo ARIMA(3,1,1)')

plot_acf(ar_est_spx_resid, lags=20)

plt.show()



plot_pacf(ar_est_spx_resid, lags=20)

plt.show()
print('Modelo AR-ARCH(3;4,0,2)')

plot_acf(am_adj_resid, lags=20)

plt.show()



plot_pacf(am_adj_resid, lags=20)

plt.show()
n_test = 100

# forecast the test set

yhat_1 = am_spx_ag.forecast(horizon = n_test)

# plot the actual variance

var = [i*0.01 for i in range(0,100)]

plt.plot(var[-n_test:])

# plot forecast variance

plt.plot(yhat_1.variance.values[-1, :])

plt.show()
n_test = 100

# forecast the test set

yhat_2 = am_adj.forecast(horizon = n_test)

# plot the actual variance

var = [i*0.01 for i in range(0,100)]

plt.plot(var[-n_test:])

# plot forecast variance

plt.plot(yhat_2.variance.values[-1, :])

plt.show()