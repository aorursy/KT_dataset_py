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
# importação das bibliotecas

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
# Renomeando as colunas

df.columns = ['Nº', 'DATA_OBS', 'PROVINCIA_ESTADO', 'PAIS_REGIAO', 'ULTIMA_ALTER', 'CONFIRMADO', 
              'MORTES', 'RECUPERADOS']        


df.head(10)
print("Size/Shape of the dataset: ",df.shape)
print("Checking for null values:\n",df.isnull().sum())
print("Checking Data-type of each column:\n",df.dtypes)

# Filtrando as base ou seja, selecionando as bases.

df_brasil = df[df['PAIS_REGIAO'] == 'Brazil']

df_brasil.head()
# Adicionando a coluna CONTAMINADOS
df_brasil['CONTAMINADOS'] = df_brasil['CONFIRMADO'] - (df_brasil['MORTES'] - df_brasil['RECUPERADOS'])

df_brasil.head(10)
df_brasil.info()
df_brasil.CONTAMINADOS .describe()
# Demostração usando o Gráfico BOx_polt

df_brasil['CONTAMINADOS'].plot.box()
# Gráfico demostrativo mostrando a curva dos contaminados

sns.pairplot(x_vars='DATA_OBS',y_vars='CONTAMINADOS', data=df_brasil, height=10)
# Separando as colunas que irão fazer árte dos modelos


df_contaminados = df_brasil.drop(columns=['Nº','PROVINCIA_ESTADO','PAIS_REGIAO','ULTIMA_ALTER','CONFIRMADO','MORTES','RECUPERADOS'])
df_contaminados.head()
df_contaminados.DATA_OBS.max()
df_contaminados
#atribuindo a coluna CONTAMINADOS no index da base
# DATA_OBS observações...

df_contaminados.index = df_contaminados['DATA_OBS']
df_contaminados.drop('DATA_OBS',inplace=True, axis=1)
df_contaminados.head()


df_contaminados.describe ()
# A mostra gráfica representa um avanço do virus Covid19.
X = [i for i in range(0, len(df_contaminados))]
X = np.reshape(X, (len(X), 1))
y = df_contaminados
modelo = LinearRegression()
modelo.fit(X, y)
modelo.coef_
tendencia = modelo.predict(X)

plt.plot(df_contaminados.values, color='blue',label='Original')
plt.title('Avanço dos contagiado pelo COVID - 19')
plt.ylabel('DATA_OBS')
plt.xlabel('CONTAMINADOS')
plt.legend('Y = 0.00028*x-2.27836')
plt.plot(tendencia,color='red', label='Tendência' )
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA

# Criando o modelo
model = ARIMA(df_contaminados, order=(5,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# Plotando os erros
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':80})

residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Resíduos", ax=ax[0])
residuals.plot(kind='kde', title='Densidade', ax=ax[1])
plt.show()
#Previsão para 60 dias....
# MODELO ARIMA................................
fig = modelo_treinado.plot_predict(start=1,end=60)
plt.title('Previssão de Contaminados, 60 dias utilizando o modelo ARIMA')
plt.ylabel('CONTAMINADOS')
plt.xlabel('DATA_OBS')
#Arma - modelos auto-regressivos de médias móveis

arma_mod = sm.tsa.ARMA(df_contaminados, (3,0)).fit(disp=False)
print(arma_mod.summary())
#Previssão para 2 meses.
fig = arma_mod.plot_predict(start=1, end=60)
plt.title('Previsão dos contaminados, utilizando o modelo ARMA')
plt.ylabel('CONTAMINADOS')
plt.xlabel('DATA_OBS')
#Previssão para 2 meses.....
# MODELO ARMA

fig = arma_mod.plot_predict(start=1, end=60)
plt.title('Previsão dos infectados para 60 dias, utilizando o modelo ARMA')
plt.ylabel('CONTAMINADOS')
plt.xlabel('DATA_OBS')
#Modelo SNaive, com sazonalidade

snaive = ARIMA(df_contaminados, order=(5, 1, 0))
snaive_treinado = snaive.fit()
print(snaive_treinado.summary())
plt.rcParams.update({'figure.figsize':(9,3)})
snaive_treinado.plot_predict(start=1,end=90)
plt.show()
# Estatísticas do modelo SARMA
modelo_sarima = sm.tsa.statespace.SARIMAX(df_contaminados, order=(1,0,1),
                                          seasonal_order=(0,1,2,7),
                                          enforce_invertibility=False).fit(max_iter=120,disp=False)

print(modelo_sarima.summary())






