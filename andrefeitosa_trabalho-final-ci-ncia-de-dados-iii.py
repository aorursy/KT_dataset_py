# Dados das Colunas

#SNo - Númeração da linha
#ObservationDate - Data da Observação 
#Province/State - Provincia/Estado
#Country/Region - Pais/Região
#Last Update - ùltima Atuliazação
#Confirmed - Casos Confirmados
#Deaths - Mortes
#Recovered - Casos Recuperados 

# Colunas Renomeadas

#Nº - Númeração da linha
#Data da Observacao - Data da Observação 
#Provincia_estado - Provincia/Estado
#Pais_regiao - Pais/Região
#Ultima_atlz - ùltima Atuliazação
#Confirmada - Casos Confirmados
#Mortes - Mortes
#Recuperada - Casos Recuperados 

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
        
        
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import warnings
warnings.filterwarnings("ignore")        
        
        

# Any results you write to the current directory are saved as output.
# importação das bibliotecas

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.ar_model import AR
from random import random
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Importação o arquivo e definição do nome

df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head(100)

# Renomeando as colunas

df.columns = ['Nº', 'DATA_OBS', 'PROVINCIA_ESTADO', 'PAIS_REGIAO', 'ULTIMA_ATLZ', 'CONFIRMADO', 
              'MORTES', 'RECUPERADOS']        


df.head(500)
# Seleção dos dados do Brasil

df_brasil = df[df['PAIS_REGIAO'] == 'Brazil']

df_brasil.head()

# Criando a coluna calculada com dados dos contaminados


df_brasil['CONTAMINADOS'] = df_brasil['CONFIRMADO'] - (df_brasil['MORTES'] - df_brasil['RECUPERADOS'])

df_brasil.head(50)

# Dados descritivos da nova coluna - Contaminados

df_brasil.CONTAMINADOS .describe()
# Gráfico demostrativo BOX SPLOT da Variavel contaminação

df_brasil['CONTAMINADOS'].plot.box()
# Gráfico demostrativo da evolução pela média dos casos contaminados

df_brasil.groupby('DATA_OBS')['CONTAMINADOS'].mean().plot()
# Gráfico evolutivo dos casos contaminados

sns.pairplot(x_vars='DATA_OBS',y_vars='CONTAMINADOS', data=df_brasil, height=10)
# Separando as colunas que irão fazer parte dos modelos


df_contaminados = df_brasil.drop(columns=['Nº','PROVINCIA_ESTADO','PAIS_REGIAO','ULTIMA_ATLZ','CONFIRMADO','MORTES','RECUPERADOS'])
df_contaminados.head()


# Verificando os formatos das colunas
#OBS: Será necessário alterar a coluna DATA_OBS para o formato datetime

df_contaminados.info()
# Convertendo a coluna DATA_OBS em datetime

df_contaminados['DATA_OBS'] = pd.to_datetime(df['DATA_OBS'])

df_contaminados.info()
df_contaminados.head()
#Atribuindo a coluna CONTAMINADOS no index da base

df_contaminados.index = df_contaminados['DATA_OBS']
df_contaminados.drop('DATA_OBS',inplace=True, axis=1)
df_contaminados.head()
df_contaminados.head()
# Plotando o gráfico de evolução para inserir a linha de regressão.

plt.plot(df_contaminados)
plt.title('Evolução')
plt.ylabel('DATA_OBS')
plt.xlabel('CONTAMINADOS')
plt.show()
# Inclusão da linha de regressão


X = [i for i in range(0, len(df_contaminados))]
X = np.reshape(X, (len(X), 1))
y = df_contaminados
modelo = LinearRegression()
modelo.fit(X, y)
modelo.coef_
tendencia = modelo.predict(X)

plt.plot(df_contaminados.values, color='blue',label='Original')
plt.title('Evolução dos contaminados pelo COVID - 19')
plt.ylabel('DATA_OBS')
plt.xlabel('CONTAMINADOS')
plt.plot(tendencia,color='red', label='Tendência' )
plt.show()
# Estatísticas do modelo ARIMA

modelo = ARIMA(df_contaminados, order=(5,1, 0))
modelo_treinado = modelo.fit()
print(modelo_treinado.summary())
#Previsão

fig = modelo_treinado.plot_predict(start=1,end=60)
plt.title('Previsão de Contaminados, 60 dias utilizando o modelo ARIMA')
plt.ylabel('CONTAMINADOS')
plt.xlabel('DATA_OBS')



#Arma - modelos auto-regressivos de médias móveis

arma_mod = sm.tsa.ARMA(df_contaminados, (3,0)).fit(disp=False)
print(arma_mod.summary())

#Previssão para 60 dias

fig = arma_mod.plot_predict(start=1, end=60)
plt.title('Previssão do COVID para 60 dias, utilizando o modelo ARMA')
plt.ylabel('CONTAMINADOS')
plt.xlabel('DATA_OBS')

#Modelo SNaive, com sazonalidade

snaive = ARIMA(df_contaminados, order=(0, 1, 0))
snaive_treinado = snaive.fit()
print(snaive_treinado.summary())
snaive_treinado.plot_predict(start=1,end=60)
plt.title('Previssão do COVID para 60 dias, utilizando o modelo NAIVE')
plt.show()
#Modelo Naive, sem sazonalidade

naive = ARIMA(df_contaminados, order=(0, 0, 0))
naive_treinado = naive.fit()
print(naive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,3)})
naive_treinado.plot_predict(start=1,end=90)
plt.title('Previssão do COVID para 90 dias, utilizando o modelo NAIVE')
plt.show()
