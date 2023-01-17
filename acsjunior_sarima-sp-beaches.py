import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from matplotlib.cbook import boxplot_stats
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from matplotlib.animation import FuncAnimation
from math import sqrt
import itertools
parser = (lambda x:datetime.datetime.strptime(x, '%Y.%m.%d')) 
df = pd.read_csv('../input/sp-beaches-update/sp_beaches_update.csv', parse_dates=['Date'])
df = df.sort_values(by=['Date'])
df=df.loc[~df['Enterococcus'].isnull()]
#remover a praia do Leste, da cidade de iguape, pois esta praia sumiu por erosão em 2012
#remover a Lagoa Prumirim, da cidade de Ubatuba, pois esta praia possui somente 3 medições
df = df.loc[df['Beach']!='DO LESTE'].loc[df['Beach']!='LAGOA PRUMIRIM']
df.info()
cidade="UBATUBA"
praia="GRANDE"
test_size=10

df_beach = df.loc[df['City']==cidade].loc[df['Beach']==praia][['Date','Enterococcus']]
df_beach.columns = ['ds', 'y']
df_beach.set_index('ds', inplace=True)

treino = df_beach.iloc[:-test_size,0:1].copy()
teste = df_beach.iloc[-test_size:,0:1].copy()

print(treino)
print(teste)
plt.figure(figsize=(18,5))
plt.title('Enterococcus na praia "'+praia+'" de '+cidade)
plt.plot(treino, color='teal')
plt.plot(teste, color='orangered')
plt.legend(['Treino','Teste'])
plt.xlabel('Data')
plt.ylabel('Enterococcus UFC/100ML')
plt.show()
future=df_beach.loc[df_beach.index[-test_size:]].index
future
def checar_estacionariedade(y, lags_plots=48, figsize=(22,8)):
    "Use Series como parâmetro"
    
    # Criando plots do DF
    #y = pd.Series(y)
    fig = plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0))
    ax3 = plt.subplot2grid((3, 3), (1, 1))
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
    y.plot(ax=ax1, figsize=figsize, color='teal')
    ax1.set_title('Série de medições')
    plot_acf(y, lags=lags_plots, zero=False, ax=ax2, color='teal');
    plot_pacf(y, lags=lags_plots, zero=False, ax=ax3, method='ols', color='teal');
    sns.distplot(y, bins=int(sqrt(len(y))), ax=ax4, color='teal')
    ax4.set_title('Distribuição dos medições')
    plt.tight_layout()
    
    print('Resultados do teste de Dickey-Fuller:')
    adfinput = adfuller(y)
    adftest = pd.Series(adfinput[0:4], index=['Teste Statistico','Valor-P','Lags Usados','Números de Observações'])
    adftest = round(adftest,4)
    
    for key, value in adfinput[4].items():
        adftest["Valores Críticos (%s)"%key] = value.round(4)
        
    print(adftest)
checar_estacionariedade(treino, lags_plots=160)
treinoLogDiff = np.log(treino).diff().dropna()
checar_estacionariedade(treinoLogDiff, lags_plots=160)

decomposition = sm.tsa.seasonal_decompose(treino, model='additive', freq=1)
fig = decomposition.plot()
plt.show()

p = range(0, 6)
d = range(0, 2)
q = range(0, 1)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
import warnings
warnings.filterwarnings("ignore")
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(treinoLogDiff,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except: 
            continue
mod = sm.tsa.statespace.SARIMAX(treino,
            order=(5, 0, 0),
            seasonal_order=(5, 1, 0, 52),
            enforce_stationarity=False,
            enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(18, 8))
plt.show()
import datetime as dt
pred_uc = results.get_forecast(steps=84)
index_date = pd.date_range(treino.index[-10], periods = 84, freq = 'W')
forecast_series = pd.Series(list(pred_uc.predicted_mean), index = index_date)
pred_ci = pred_uc.conf_int()
ax = treino['2019':].plot(label='treino', figsize=(14, 4))
#print(pred_uc.predicted_mean)
forecast_series.plot(ax=ax, label='predito')
teste.plot(ax=ax, label='teste')
ax.fill_between(forecast_series.index,
 pred_ci.iloc[:, 0],
 pred_ci.iloc[:, 1], color='k', alpha=.1)
ax.set_xlabel('Data')
ax.set_ylabel('Enterococos')
plt.legend()
plt.show()
results.summary()

from sklearn.metrics import mean_squared_error
#pred = results.predict('2020-08-24','2020-09-28',exog = teste)[1:]
#forecast_series
print(forecast_series['2020-07-20':'2020-09-28'])
print(df_beach['2020-07-27':'2020-09-28'])
print('ARIMAX model MSE:{}'.format(mean_squared_error(df_beach['2020-07-27':'2020-09-28'],forecast_series['2020-07-20':'2020-09-28'])))
print(df_beach['2020-07-20':'2020-09-28'].to_numpy().reshape((10)))
print(forecast_series['2020-07-20':'2020-09-28'].to_numpy())
pd.DataFrame({'test':df_beach['2020-07-20':'2020-09-28'].to_numpy().reshape((10)),'pred':forecast_series['2020-07-20':'2020-09-28'].to_numpy()}).plot();plt.show()
results.save('my_model.pkl')