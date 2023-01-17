import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
df = pd.read_csv("../input/daily-climate-time-series-data/DailyDelhiClimateTest.csv")

#Obs: eu usei o dataset de teste sem perceber mas no final ficou mais didático :D
df.head()
df.info()
df.date = pd.DatetimeIndex(df.date.values)

df = df.set_index('date')

df.head()
df.describe()
df_exemple = pd.read_csv("../input/daily-climate-time-series-data/DailyDelhiClimateTest.csv")

df_exemple.date = pd.to_datetime(df_exemple.date)

df_exemple = df_exemple.set_index('date')
df_exemple.resample('M').mean()
df_exemple.resample('H').ffill()
df_exemple.resample('H').bfill()
df.plot(subplots=True, figsize=(10,8))
df.meantemp.hist(grid=False, bins=10, )

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df.meantemp)

result.plot()

plt.show()
# import matplotlib.dates as mdates



fig, ax = plt.subplots(figsize=(10,7))

plt.subplots_adjust(hspace=0.5)



ax0 = plt.subplot(411)

plt.plot(result.observed)

ax0.set_title('observado')



ax1 = plt.subplot(412)

plt.plot(result.trend)

ax1.set_title('tendência')



ax2 = plt.subplot(413)

plt.plot(result.seasonal)

ax2.set_title('sazonalidade')



ax3 = plt.subplot(414)

plt.plot(result.resid)

ax3.set_title('resíduo')

fig.autofmt_xdate()

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.stattools import acf, pacf



print('Resultados - Dickey Fuller Teste:')

adftest = adfuller(df['meantemp'])



out = pd.Series(adftest[0:4], index=['Teste','p-valor','Lags','Número de observações usadas'])

for key,value in adftest[4].items():

    out['Valor crítico (%s)'% key] = value

print(out)
dfdiff = df.meantemp.diff()

dfdiff = dfdiff.dropna()

plt.title("Primeira diferenciação")

dfdiff.plot()

plt.show()
print('Results of Dickey Fuller Test: Diferenciação')

dftest = adfuller(dfdiff, autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
#Determine rolling statistics

mediamovel = df['meantemp'].rolling(window=7).mean() #window = 7 diz que a janela usada para média móvel é de 7 observações (no caso 7 dias)

desviomovel = df['meantemp'].rolling(window=7).std()



fig, ax = plt.subplots(figsize=(10,7))





orig = plt.plot(df['meantemp'], color='blue', label='Observado')

media = plt.plot(mediamovel, color='red', label='Média Móvel')

desvio = plt.plot(desviomovel, color='black', label='Desvio Padrão Móvel')

plt.legend(loc='best')

plt.title('Estatísticas de rolagem')

fig.autofmt_xdate()



plt.show(block=False)
temp_log = np.log(df['meantemp'])

fig, ax = plt.subplots()

plt.plot(temp_log)

plt.title("Transformação logarítma")

ax.xaxis_date()

fig.autofmt_xdate()
print('Results of Dickey Fuller Test:')

dftest = adfuller(temp_log, autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
#Determine rolling statistics

rolmean_log = temp_log.rolling(window=7).mean() 

rolstd = temp_log.rolling(window=7).std()



fig, ax = plt.subplots(figsize=(8,6))



orig = plt.plot(temp_log, color='blue', label='Transformação logarítmica')

mean = plt.plot(rolmean_log, color='red', label='Média Móvel da transformação')

plt.legend(loc='best')

plt.title('Estatísticas de rolagem - Log')

ax.xaxis_date()

fig.autofmt_xdate()

plt.show(block=False)
log_menos_media = temp_log - rolmean_log

#Remove NAN values

log_menos_media.dropna(inplace=True)
#Determine rolling statistics

rolmean = log_menos_media.rolling(window=7).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level

rolstd = log_menos_media.rolling(window=7).std()



fig, ax = plt.subplots(figsize=(8,6))



orig = plt.plot(log_menos_media, color='blue', label='Observado')

mean = plt.plot(rolmean, color='red', label='Média Móvel')

std = plt.plot(rolstd, color='black', label='Desvio Padrão Móvel')

plt.legend(loc='best')

plt.title('Estatísticas de rolagem')



ax.xaxis_date()

fig.autofmt_xdate()

plt.show(block=False)
print('Results of Dickey Fuller Test: Log menos média')

dftest = adfuller(log_menos_media, autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)
from scipy.stats import boxcox

meantemp_bcx, lam = boxcox(df['meantemp'])

print('Lambda: %f' % lam)



fig, ax = plt.subplots()

# line plot

fig.suptitle("BoxCox resultados")

plt.subplot(211)



plt.plot(meantemp_bcx)

# histogram

plt.subplot(212)

plt.hist(meantemp_bcx)

plt.show()



print('Results of Dickey Fuller Test:')

dftest = adfuller(meantemp_bcx, autolag='AIC')



dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

for key,value in dftest[4].items():

    dfoutput['Critical Value (%s)'%key] = value

    

print(dfoutput)