import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Lendo o arquivo contendo todos os voos
voos = pd.read_parquet('/kaggle/input/voo-regular-ativo-vra-anac/arquivo_limpo.parquet.gzip')
# Lendo o arquivo de descrição de siglas de aeroportos
aeroportos = pd.read_excel('../input/voo-regular-ativo-vra-anac/glossario_de_aerodromo.xls', header=3)
aeroportos = aeroportos.drop('Unnamed: 0', axis=1)
# Amostra aleatória de 10 voos
voos.sample(10)
# Verificando 10 primeiros aeroportos da lista
aeroportos.head(10)
# Verificando estrutura da tabela de voos
voos.info()
# União das tabelas, primeiro aeroporto de origem e depois o de destino
voos = voos.merge(aeroportos, how='left', left_on='Aeroporto Origem', right_on='Sigla OACI')
voos = voos.merge(aeroportos, how='left', left_on='Aeroporto Destino', right_on='Sigla OACI', suffixes=('_origem', '_destino'))

# Eliminando voos externos do Brasil
voos = voos[(voos['País_origem']=='BRASIL')&(voos['País_destino']=='BRASIL')]

# Resetando o indice após a remoção de voos desnecessários
voos = voos.reset_index(drop=True)
voos['Situação'] = voos['Situação'].str.lower()
voos['País_origem'] = voos['País_origem'].str.lower()
voos['País_destino'] = voos['País_destino'].str.lower()
voos['Cidade_origem'] = voos['Cidade_origem'].str.lower()
voos['Cidade_destino'] = voos['Cidade_destino'].str.lower()
# Verificando as situações que existem na tabela
voos['Situação'].value_counts()
# Calculando os atrasos na partida e chegada
voos['Atraso Partida'] = voos['Partida Real'] - voos['Partida Prevista']
voos['Atraso Chegada'] = voos['Chegada Real'] - voos['Chegada Prevista']
# Classificando aqueles acima de 30 minutos
voos['Atrasados Partida'] = 0
voos.loc[voos['Atraso Partida'] > datetime.timedelta(minutes=30), 'Atrasados Partida'] = 1

voos['Atrasados Chegada'] = 0
voos.loc[voos['Atraso Chegada'] > datetime.timedelta(minutes=30), 'Atrasados Chegada'] = 1
# Verificando o total
print('Atrasados Partida: ', voos['Atrasados Partida'].sum())
print('Atrasados Chegada: ', voos['Atrasados Chegada'].sum())
# Colocando como indice o horário da partida prevista para gerar uma série temporal
voos = voos.set_index('Partida Prevista')
# Todos os voos atrasados na partida por mês (2000-2020)
atrasos_por_mes_partida = voos['Atrasados Partida'].groupby(pd.Grouper(freq='M')).sum()
# atrasos_por_mes_partida.plot(figsize=(20,10))
# Todos os voos atrasados na chegada agrupados por mes (2000-2020)
atrasos_por_mes_chegada = voos['Atrasados Chegada'].groupby(pd.Grouper(freq='M')).sum()
# atrasos_por_mes_chegada.plot(figsize=(20,10))
# Todos os voos cancelados
cancelados = voos[voos['Situação']=='cancelado']['Situação'].groupby(pd.Grouper(freq='M')).count()
# cancelados.plot(figsize=(20,10))
atrasos_por_mes_partida.index[172]
# Unificação dos gráficos dos cálculos acima
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

plt.figure(figsize=(20,10))

atrasos_por_mes_partida.plot()
atrasos_por_mes_chegada.plot()
cancelados.plot(label=("Cancelados"))

plt.annotate('Início Apagão Controladores Aéreos',
             (atrasos_por_mes_partida.index[80], atrasos_por_mes_partida[80]),
             xytext=(5, -80), 
             textcoords='offset points',
             arrowprops=dict(arrowstyle='-|>'))

plt.annotate('Copa do Mundo',
             (atrasos_por_mes_partida.index[173], atrasos_por_mes_partida[173]),
             xytext=(10, -25), 
             textcoords='offset points',
             arrowprops=dict(arrowstyle='-|>'))


plt.title('Situação por Mês')
plt.ylabel('Atrasos')
plt.xlabel('Ano/Mês')
plt.legend()
plt.show()
atrasos_por_mes_bsb_origem = voos.loc[voos['Cidade_origem']=='brasília','Atrasados Partida'].groupby(pd.Grouper(freq='M')).sum()
atrasos_por_mes_bsb_origem.plot(figsize=(20,10))
# Exemplo de filtro de uma cidade destino: Brasília
atrasos_por_mes_bsb_destino = voos.loc[voos['Cidade_destino']=='brasília','Atrasados Partida'].groupby(pd.Grouper(freq='M')).sum()
atrasos_por_mes_bsb_destino.plot(figsize=(20,10))
# Exemplo de filtro de uma cidade destino, com uma data inicial: Brasília, a partir de 2010

atrasos_por_mes_bsb_destino = voos.loc[(voos['Cidade_destino']=='brasília')&(voos['Partida Real']>'2010-01-01'),'Atrasados Partida'].groupby(pd.Grouper(freq='M')).sum()
atrasos_por_mes_bsb_destino.plot(figsize=(20,10))
# Exemplo de filtro de uma cidade origem, com uma data inicial: Brasília, a partir de 2010
atrasos_por_mes_bsb_origem = voos.loc[(voos['Cidade_origem']=='brasília')&(voos['Partida Real']>'2010-01-01'),'Atrasados Partida'].groupby(pd.Grouper(freq='M')).sum()
atrasos_por_mes_bsb_origem.plot(figsize=(20,10))
from sklearn.model_selection import TimeSeriesSplit

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.arima_model import ARMA
# Cálculo ACF e PACF 

# Partida

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 3))
plot_acf(atrasos_por_mes_partida, ax=ax, lags=12)

fig, ax = plt.subplots(figsize=(12, 3))
plot_pacf(atrasos_por_mes_partida, ax=ax, lags=12)
# Chegada

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(12, 3))
plot_acf(atrasos_por_mes_chegada, ax=ax, lags=12)

fig, ax = plt.subplots(figsize=(12, 3))
plot_pacf(atrasos_por_mes_chegada, ax=ax, lags=12)
