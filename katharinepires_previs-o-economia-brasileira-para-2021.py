import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go
ipca = pd.read_csv('../input/economia-brasileira/ipca.csv', sep = ';')

selic = pd.read_csv('../input/economia-brasileira/taxa_selic.csv', sep = ';')
ipca
selic
ipca = ipca.rename(columns={'13522 - Índice nacional de preços ao consumidor - amplo (IPCA) - em 12 meses - %': 'IPCA'})

selic = selic.rename(columns={'Meta para taxa Selic': 'Selic'})
ipca
selic
fig = go.Figure(go.Scatter(x = selic['DateTime'], y = selic['Selic'], name = 'Taxa Selic'))

fig.update_layout(title = 'Taxa Selic ao longo de 2020')

fig.show()
fig = go.Figure(go.Scatter(x = ipca['Data'], y = ipca['IPCA'], name = 'IPCA'))

fig.update_layout(title = 'IPCA ao longo de 2020')

fig.show()
fig = go.Figure(go.Scatter(x = selic['DateTime'], y = selic['Selic'], name = 'Taxa Selic'))

fig.add_trace(go.Scatter(x = ipca['Data'], y = ipca['IPCA'], name = 'IPCA'))

fig.update_layout(title = 'IPCA & Selic no ano de 2020')

fig.show()
pred_selic = selic.iloc[:182 , :]

pred_selic
pred_ipca = ipca.iloc[:6 , :]

pred_ipca
!pip install pmdarima
from pmdarima.arima import auto_arima
pred_selic.loc[:, "DateTime"] = pred_selic.DateTime.map(pd.Timestamp)

pred_ipca.loc[:, "Data"] = pred_ipca.Data.map(pd.Timestamp)
pred_selic2 = pred_selic.Selic

pred_ipca2 = pred_ipca.IPCA
pred_selic = selic.iloc[:182 , :]

pred_selic.index = pred_selic.DateTime

pred_ipca = ipca.iloc[:6 , :]

pred_ipca.index = pred_ipca.Data
modelo_selic = auto_arima(pred_selic2)

previsao_selic = modelo_selic.predict(365)
sns.set()

plt.figure(figsize=(15,8))

plt.plot(pd.date_range('2020-07-01', '2021-06-30'), previsao_selic)

plt.ylabel('')

plt.xlabel('')

plt.title('Selic nos próximos 12 meses', fontsize = 20);

plt.show()
modelo_ipca = auto_arima(pred_ipca2)

previsao_ipca = modelo_ipca.predict(365)
sns.set()

plt.figure(figsize=(15,8))

plt.plot(pd.date_range('2020-07-01', '2021-06-30'), previsao_ipca)

plt.ylabel('')

plt.xlabel('')

plt.title('IPCA nos próximos 12 meses', fontsize = 20);

plt.show()
fig = go.Figure(go.Scatter(x = pd.date_range('2020-07-01', '2021-06-30'), y = previsao_selic, name = 'Previsão da Meta para a Taxa Selic'))

fig.add_trace(go.Scatter(x = pd.date_range('2020-07-01', '2021-06-30'), y = previsao_ipca, name = 'Previsão para o IPCA'))

fig.update_layout(title = 'Economia Brasileira em 2021')

fig.show()