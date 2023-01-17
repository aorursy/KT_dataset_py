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
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv', 
                      parse_dates=['Last Update'])
df['Data'] = df['Last Update'].dt.date.astype(str)
df.head()
df.info()
# obtendo base apenas do Brazil

df_Brazil= df[df['Country/Region']=='Brazil']
df_Brazil.shape
df_Brazil.describe()
df_Brazil.info()

df_Brazil['Data'] = df_Brazil['Last Update'].dt.date.astype(str)
df_Brazil.info()
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df_Brazil = df_Brazil.drop(columns=['SNo','Province/State'])
df_Brazil.head()
#convertendo Data par formato datetime
df_Brazil['Data']=pd.to_datetime(df_Brazil['Data'])
df_Brazil = df_Brazil.sort_values(by='Data')
df_Brazil.info()
df_Brazil.index =df_Brazil['Data']

#incluindo coluna de infectados (casos confirmando menos mortos e recuperados)
df_Brazil['Infectados'] = df_Brazil['Confirmed']- (df_Brazil['Deaths'] + df_Brazil['Recovered'])
df_Brazil.head()
df_Brazil2 = df_Brazil.iloc[:,7:9]
df_Brazil2.head()
df_Brazil2['Infectados'] = df_Brazil2['Infectados'].values.astype(np.int64)
df_Brazil2.dtypes
#criando serie de infectado
serie=df_Brazil2
serie.info(), serie.head()


media_movel = serie.rolling(window =7).mean()

plt.plot(serie)
plt.plot(media_movel, color ='green')
#iniciando modelo ARIMA 
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 

print ("p-value:", adfuller (serie.dropna ()) [1])
fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(311)
fig = plot_acf(serie, ax=ax1,
               title='Autocorrelação na série original') 
ax2 = fig.add_subplot(312)
fig = plot_acf(serie.diff().dropna(), ax=ax2, 
               title='Diferença de primeira ordem')
ax3 = fig.add_subplot(313)
fig = plot_acf(serie.diff().diff().dropna(), ax=ax3, 
               title='Diferença de segunda ordem')
plot_pacf(serie.diff().dropna(), lags=40)
plot_acf(serie.diff().dropna())
modelo = ARIMA(serie, order=(1,1,1))
modelo_treinado = modelo.fit()
modelo_treinado.plot_predict(1,256)
