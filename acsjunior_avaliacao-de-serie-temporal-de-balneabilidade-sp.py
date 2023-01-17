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
from matplotlib.animation import FuncAnimation
from math import sqrt
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
test_size=5

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
checar_estacionariedade(df_beach)
treino = np.log(df_beach).diff().dropna()
checar_estacionariedade(treino)
# Treinando o modelo
modelo = ARIMA(treino, order=(4,1,1)).fit()
pred_treino = modelo.predict()
# Base de previsão com diferenciação
pred_treino[:5]
# Voltando para a base de preços em R$/m3
pred_treino[0] += treino.iloc[0,0]
pred_treino = np.cumsum(pred_treino)
pred_treino.head()
treino['pred y'] = pred_treino
treino.dropna(inplace=True)
treino.head()
treino.plot(figsize=(18,6), 
            title=' Enterococcus - Real vs Previsto na base de Treino',
           color=['Teal','orangered'])
plt.ylabel('Enterococcus UFC/100ML')
plt.show()