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
#Importações de todas as bibliotecas necessárias



import numpy as np

import pandas as pd

from dateutil.parser import parse 

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
# Carregando a base de dados

BDTotal = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', parse_dates=['Last Update'])
# Criando um dataframe e fazendo o filtro dos dados somente por Brasil

df = BDTotal[BDTotal['Country/Region'] == 'Brazil']
#Criando o campo contaminados, que é composto dos confirmados menos os mortos e recuperados

df['Doentes'] = df['Confirmed'] - df['Deaths'] - df['Recovered']
# Visualização geral dos dados

df
# tamanhos da base

df.shape
# tipos de variáveis

df.info()
# Explorando

df.describe()
df.head(10)
# Transformar o ObservationDate



df['ObservationDate'] = pd.to_datetime(df['ObservationDate'])
# Separar apenas as colunas que serão utilizadas



data = df[['ObservationDate', 'Doentes']]
# lendo a base data

data
# Conferir as variáveis de data

data.info()
# Setando a coluna ObservationDate como index



data.set_index('ObservationDate',inplace=True)
# Tranformar o dataset em série temporal



ts = data['Doentes']
# Estabelecendo análise por gráfico



plt.plot(ts)
# Realizar transformação em logaritmo



ts_log = np.log(ts)

plt.plot(ts_log)
# Dropar valores "não válidos".



ts_log.drop(ts_log.index[[0]],inplace=True)
# Dropar valores Na



ts_log.dropna()
#Importando modelo ARIMA



from statsmodels.tsa.arima_model import ARIMA
# Fazendo o modelo ARIMA



from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.arima_model import ARIMA
# Criando o modelo

model = ARIMA(df.Doentes, order=(1,1,2))

model_fit = model.fit(disp=0)

print(model_fit.summary())
# Plotando os erros

plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':120})



residuals = pd.Dataframe(model_fit.resid)

fig, ax = plt.subplots(1,2)

residuals.plot(title="Resíduos", ax=ax[0])

residuals.plot(kind='kde', title='Densidade', ax=ax[1])

plt.show()