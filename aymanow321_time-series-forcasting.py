import numpy as np

import pandas as pd

from pandas_datareader import data as wb

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 15,6

from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.tsa.stattools import acf , pacf

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
donnees = pd.read_csv('../input/donnees/donnees.csv')

donnees['Date'] = pd.to_datetime(donnees['Date'])

donnees = donnees.set_index('Date')



plt.plot(donnees)
donnees.head()
donnees.dtypes
donnees = np.log(donnees)
moyenne_mobile = donnees.rolling(window = 12).mean()



plt.xlabel('Date')

plt.title('Valeur')

Valeur = plt.plot(donnees, color = 'blue',label = 'Valeur des données')

moyenne_mobile = plt.plot(moyenne_mobile ,color = 'red',label = 'Moyenne Mobile')

plt.legend(loc = 'best')

#plt.show(block = False ) 
decomposition = seasonal_decompose(donnees,freq=12)

tendance = decomposition.trend

saisonnalite = decomposition.seasonal

residu = decomposition.resid



plt.subplot(411)

plt.plot(donnees , label = 'donnees')

plt.legend(loc = 'best')

plt.subplot(412)

plt.plot(tendance , label = 'tendance')

plt.legend(loc = 'best')

plt.subplot(413)

plt.plot(saisonnalite , label = 'saisonalité')

plt.legend(loc = 'best')

plt.subplot(414)

plt.plot(residu , label = 'residual')

plt.legend(loc = 'best')



donnees_diff = donnees - donnees.shift()

donnees_diff.dropna(inplace = True)

donnees_diff.head()
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.subplot(211)

plot_acf(donnees_diff,lags = 13, ax=plt.gca())

plt.subplot(212)

plot_pacf(donnees_diff,lags = 13, ax=plt.gca())

plt.show()
model = ARIMA(donnees , order = (2,1,2))

results_ARIMA = model.fit(disp = 1)



plt.plot( donnees_diff )

plt.plot( results_ARIMA.fittedvalues, color='red')

plt.title('RSS = %.4F'% sum((results_ARIMA.fittedvalues - donnees_diff['Valeur'])**2))
#results_ARIMA.fittedvalues.head()
prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues , copy = True)

#print(prediction_ARIMA_diff.head())
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()

#print(prediction_ARIMA_diff_cumsum.head())
predictions_ARIMA_log = pd.Series(donnees['Valeur'] , index = donnees.index )

predictions_ARIMA_log = predictions_ARIMA_log.add(prediction_ARIMA_diff_cumsum , fill_value = 0)

#predictions_ARIMA_log.head()
predictions_ARIMA = np.exp(predictions_ARIMA_log)

donnees = np.exp(donnees)



plt.xlabel('Date')

plt.title('Données et leur modélisation')

Valeur = plt.plot(donnees, color = 'blue',label = 'Valeur des données')

moyenne_mobile = plt.plot(predictions_ARIMA ,color = 'red',label = 'valeur du modél')

plt.legend(loc = 'best')
results_ARIMA.plot_predict(1,donnees.size + 24) 

plt.show()