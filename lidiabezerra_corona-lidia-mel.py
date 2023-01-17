# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
###Importando as bibliotecas
#Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.dates as mdates
from datetime import date, timedelta
import statsmodels.api as sm
import statsmodels.api as tsa
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from matplotlib import pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from random import random
####IMPORTANDO O CONJUNTO DE DADOS
corona = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
###Estudando a base principalmente casos confirmados
corona.shape

###Observa-se qeue existem 19729 casos e 8 colunas/variáveis

###Verificando as variáveis na forma transposta
corona.head().T
###Coletando informações dos dados
corona.info()
##Verificando se existem MISSING VALUES. Foi também calculado o percentual de missing cases

feat_missing = []

for f in df.columns:
    missings = corona[f].isnull().sum()
    if missings > 0:
        feat_missing.append(f)
        missings_perc = missings/corona.shape[0]
        
###Verificando o percentual de missing
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
    
###Verificando quantas variáveis apresentam os casos faltosos

print()
print('In total, there are {} variables with missing values'.format(len(feat_missing)))
###Tratando os missing values
###Tratando os missing values
corona = corona.fillna(method='ffill')
corona = corona.fillna(method='bfill')
###COnfirmando se os missing values foram imputados
feat_missing = []

for f in df.columns:
    missings = corona[f].isnull().sum()
    if missings > 0:
        feat_missing.append(f)
        missings_perc = missings/corona.shape[0]
        
###Verificando o percentual de missing
        print('Variable {} has {} records ({:.2%}) with missing values'.format(f, missings, missings_perc))
    
###Verificando quantas variáveis apresentam os casos faltosos

print()
print('In total, there are {} variables with missing values'.format(len(feat_missing)))


####É possível verificar que os MISSING VALUES foram impultados com sucesso.
###Agora verificando a estatística descritiva das variáveis quantitativas
print(corona.describe().T)
corona.head()
#Renomear colunas
corona = corona.rename(columns={"ObservationDate":"data", "Country/Region": "pais", "Province/State": 
                              "estado","Confirmed":"confirmados", "Deaths": "mortes","Recovered":"recuperados"})
corona.head()
####FrequNcia de casos
corona[corona['date'] != corona['Last Update']]['country'].value_counts()
###Gráfico
plt.figure(figsize=(15,6))
plt.title('Number of Province/State were affected in Each Country')
plt.xticks(rotation=90)
prv_lst = corona.groupby(['Country'])['Province/State'].nunique().sort_values(ascending=False)
prv_lst.plot(kind='bar')
plt.tight_layout()
###Verificando os nomes das colunas
corona.columns
###Excluíndo a coluna SNo
corona = corona [['date', 'country', 'Last Update', 'confirm', 'death',
       'recover']]
corona.head()
###Checando missing values
corona.isna().sum()
corona.describe()
corona.columns
###Número de casos por data
corona.head()
###Verificando o número de casos confirmados, mortes e recuperados no geral
corona.groupby('date')['confirm','death', 'recover'].sum()
###Verificando o número de casos confirmados, mortes e recuperados no geral, analisando pela frequencia acumulada. 
###Talvez tirar
corona.groupby('date')['confirm','death', 'recover'].max()
####Verificando número de casos, mortes, recuperação POR DIA
corona_dia = corona.groupby('date')['confirm','death', 'recover'].sum()
###Verificando a contagem
corona_dia.head()
###Coferindo o máximo e o mínimo 
corona_dia.describe()
##Verificando qual país mais teve casos e mortos confirmados
corona_dia['confirm'].sum()
##Verificando qual país mais teve casos e mortos confirmados
corona_dia['confirm'].max()
##Verificando qual país mais teve casos e mortos confirmados
corona_dia['confirm'].min()
###Verificando a data com maior número de casos
corona_dia['confirm'].idxmax()
corona_dia.columns
#Número de casos por país
corona.groupby(['country'])['confirm', 'death', 'recover'].max()
####Contagem do maior para o menor
corona['country'].value_counts()
##Gráfico
corona['country'].value_counts().plot(kind='bar',figsize=(60,10))



##Identificando o total de países/região afetados
len(corona['country'].unique())

###Fato curioso, até o dia 27 de Fevereiro eram apenas 53 países 

####INiciando a análise série temporal
corona.head()
corona_dia.head()
###Criando uma cópia do dataset
corona2 = corona
##Identificando os países/região afetados
corona['country'].unique()
##Salvando a base tratada
corona.to_csv("corona_dados_limpos.csv")
##Importando a biblioteca
import datetime as dt
##Criando uma nova variável para a análises série temporal
corona['cases_date']=pd.to_datetime(corona2['date'])
corona2.dtypes
corona['cases_date'].plot(figsize=(20,10))
ts = corona2.set_index('cases_date')
#Analisando a nova variável case-date na base
ts
##TEstando a seleção de um mês específico, exemplo Janeiro de 2020
ts.loc['2020-01']

##TEstando a seleção de um mês específico, exemplo Janeiro de 2020 em duas datas específicas
ts.loc['2020-02-01':'2020-03-01']

###INICIANDO AS ANÁLISES NO BRASIL
corona_brasil = corona[corona['country']=='Brazil']

corona_brasil.head()
##Verificando número de casos 
corona_brasil.shape
##É possível observar que apenas 57 observações
#Filtrar as colunas 
corona_brasil.drop(columns=[ 'death', 'recover', 'Last Update', 'country', 'cases_date'], inplace=True, axis=1)
corona_brasil.head()
corona_brasil.index = corona_brasil['date']
corona_brasil.drop('date',inplace=True, axis=1)
corona_brasil.head()
plt.figure(figsize=(40,20))
plt.plot(corona_brasil)
plt.xlabel('Data')
plt.ylabel('Confirmado')
plt.title('Data vs casos confirmados')
plt.xticks(rotation='vertical')
plt.show()
mean = corona_brasil.rolling(window=12).mean()
m_std = corona_brasil.rolling(window=12).std()

ori = plt.plot(corona_brasil, color='blue', label='Original')
mm = plt.plot(mean, color='red',label='Média')
std = plt.plot(m_std, color='black', label='Desvio')
plt.legend(loc='best')
plt.title('Evolução dos casos de coronavírus')
  
plt.show()

result = adfuller(corona_brasil['confirm'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
   
    print('\t{}: {}'.format(key, value))

#Número de casos confirmados durante o periodo de Fevereiro a Abril 
per = np.log(corona_brasil).diff().dropna()
per.plot()
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(corona_brasil)
corona_brasil.shift(1)
modelo = ARIMA(corona_brasil, order=(0, 1, 1)).fit()
print(modelo.summary())
##Modelo preditivo para COVID-19
fig,ax = plt.subplots(figsize=(8,6))
fig = modelo.plot_predict(start='02-28-2020',end='04-18-2020',ax=ax)
plt.title('Previsão COVID_19')
plt.ylabel('Confirmados')
plt.xlabel('Data')
legend = ax.legend(loc='upper left')

#Modelo Naive sem sazonalidade
naive = ARIMA(corona_brasil, order=(0, 0, 0))
naive_treinado = naive.fit()
print(naive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,3)})
naive_treinado.plot_predict(start=1,end=230)
plt.show()
#Modelo naive com sazonalidade

snaive = ARIMA(corona_brasil, order=(0, 1, 0))
snaive_treinado = snaive.fit()
print(snaive_treinado.summary())
#Treinando o modelo com sazonalidade
snaive_treinado.plot_predict(start=1,end=230)
plt.show()
mod = sm.tsa.statespace.SARIMAX(timeseries=['y'],
                               trend=['x'],
                               order=(1,0,1),
                               seasonal_order=(0, 1, 2, 7),
                               enforce_invertibility = False)

results=mod.fit()
results.summary()
