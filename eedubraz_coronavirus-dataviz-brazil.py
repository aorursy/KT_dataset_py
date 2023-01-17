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
import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

print(df.head())

print('\n')

print(df.info())
df['data_tratada'] = pd.to_datetime(df.ObservationDate)
# Verifica o preenchimento da coluna Country/Region

df['Country/Region'].sort_values().unique()
replace_country = {' Azerbaijan':'Azerbaijan',"('St. Martin',)":'St. Martin'}

df['Country/Region'] = df['Country/Region'].replace(replace_country)
# Criando dataset agrupado por Country/Region e data

dfDia = df.groupby(['Country/Region', 'data_tratada']).agg('sum').reset_index().drop('SNo', axis=1)

dfDia.sort_values(by=['Country/Region','data_tratada'], inplace=True)

dfDia.head()
# Criando dicionário com a data do primeiro caso de coronavirus confirmado em cada país

first_confirmed = {}

for c in dfDia['Country/Region'].unique():

    first = dfDia[(dfDia['Country/Region'] == c) & (dfDia['Confirmed'] > 0)]['data_tratada'].min()

    first_confirmed[c] = first
def calc_dias(dt, country):

    '''

    Função para calcular a quantidade de dias passados desde a data do primeiro caso de coronavírus confirmado

    '''

    dt_conf = first_confirmed.get(country)

    if dt >= dt_conf:

        return (dt - dt_conf).days

    else:

        return np.nan
# Aplica a função calc_dias em todo o dataset diário

dfDia['dias_corridos'] = dfDia.apply(lambda x: calc_dias(x['data_tratada'], x['Country/Region']), axis=1)
# Verifica quais países possuem atualização no dia 15/3

dfDia.groupby('Country/Region')['data_tratada'].agg('max').value_counts()
# Verificando os últimos valores registrados por país

dfDia.groupby('Country/Region')['Confirmed'].agg('max').sort_values(ascending=False)
# Criando dataframe sem a China

dfDia2  = dfDia[ dfDia['Country/Region'] != 'Mainland China']
# Variável com a quantidade de dias a ser mostrada no gráfico. ùltima data de atualização no Brasil + 3 dias

qtd_dias_brasil = int(dfDia2[ dfDia2['Country/Region'] == 'Brazil']['dias_corridos'].max())

qtd_dias_analise = qtd_dias_brasil + 30
# Criando dataframes para plotagem

dados_brasil = dfDia2[ dfDia2['Country/Region'] == 'Brazil']

dados_italia = dfDia2[ dfDia2['Country/Region'] == 'Italy']

dados_us = dfDia2[ dfDia2['Country/Region'] == 'US']

dados_mundo = dfDia2[ dfDia2['Country/Region'] != 'Brazil']
plt.style.use('ggplot')



plt.figure(figsize=(13,6))

plt.title(f'Evolução de Casos Confirmados (COVID-19) no Mundo\n({qtd_dias_analise} dias corridos após a primeira confirmação em cada país)\n\n', size=20)

mundo = plt.plot(dados_mundo['dias_corridos'],dados_mundo['Confirmed'], color = 'lightgray', label = 'Outros países')

us = plt.plot(dados_us['dias_corridos'],dados_us['Confirmed'], color = 'blue', marker='o', linewidth=2, label='USA')

italia = plt.plot(dados_italia['dias_corridos'],dados_italia['Confirmed'], color = 'green', marker='o', linewidth=2, label='Itália')

brasil = plt.plot(dados_brasil['dias_corridos'],dados_brasil['Confirmed'], color = 'red', marker='o', linewidth=2, label='Brasil')

plt.xticks(range(0,qtd_dias_analise+1), fontsize=9)

# Rótulo Brasil

plt.text(qtd_dias_brasil+0.5, dfDia2[ dfDia2['Country/Region'] == 'Brazil']['Confirmed'].max(), int(dfDia2[ dfDia2['Country/Region'] == 'Brazil']['Confirmed'].max()), fontsize=15, color='red')

# Rótulo Italia

plt.text(28+0.4, int(dados_italia[ dados_italia['dias_corridos'] == 28.0 ]['Confirmed']) , int(dados_italia[ dados_italia['dias_corridos'] == 28.0 ]['Confirmed']), fontsize=15, color='green')

# Rótulo USA

plt.text(46-0.4, int(dados_us[ dados_us['dias_corridos'] == 48.0 ]['Confirmed']) , int(dados_us[ dados_us['dias_corridos'] == 48.0 ]['Confirmed']), fontsize=15, color='blue')

plt.xlim([0,qtd_dias_analise])

plt.ylim([0,1000])

plt.xlabel('\nDias Corridos', fontsize=15)

plt.ylabel('Casos Confirmados\n', fontsize=20)

plt.legend(fontsize=12)

plt.text(0.2,1010,'Dados atualizados até o dia {}.'.format(dfDia2["data_tratada"].max().strftime("%d/%m/%Y")))

plt.text(0,-150,'Criado por:\nEduardo Braz Rabello')
plt.style.use('ggplot')



plt.figure(figsize=(15,10))

plt.yscale('log')

plt.title(f'Evolução de Casos Confirmados (COVID-19) no Mundo\n({qtd_dias_analise} dias corridos após a primeira confirmação em cada país)\n', size=20)

mundo = plt.plot(dados_mundo['dias_corridos'],dados_mundo['Confirmed'], color = 'lightgray', label = 'Outros países')

us = plt.plot(dados_us['dias_corridos'],dados_us['Confirmed'], color = 'blue', marker='o', linewidth=2, label='USA')

italia = plt.plot(dados_italia['dias_corridos'],dados_italia['Confirmed'], color = 'green', marker='o', linewidth=2, label='Itália')

brasil = plt.plot(dados_brasil['dias_corridos'],dados_brasil['Confirmed'], color = 'red', marker='o', linewidth=2, label='Brasil')

#plt.xticks(range(0,60,5), fontsize=9)

# Rótulo Brasil

plt.text(qtd_dias_brasil+0.5, dfDia2[ dfDia2['Country/Region'] == 'Brazil']['Confirmed'].max(), int(dfDia2[ dfDia2['Country/Region'] == 'Brazil']['Confirmed'].max()), fontsize=15, color='red')

# Rótulo Italia

#plt.text(28+0.4, int(dados_italia[ dados_italia['dias_corridos'] == 28.0 ]['Confirmed']) , int(dados_italia[ dados_italia['dias_corridos'] == 28.0 ]['Confirmed']), fontsize=15, color='green')

# Rótulo USA

#plt.text(46-0.4, int(dados_us[ dados_us['dias_corridos'] == 48.0 ]['Confirmed']) , int(dados_us[ dados_us['dias_corridos'] == 48.0 ]['Confirmed']), fontsize=15, color='blue')

plt.xlabel('\nDias Corridos', fontsize=15)

plt.ylabel('Casos Confirmados\n', fontsize=20)

plt.legend(fontsize=12)

#plt.text(0.2,1010,'Dados atualizados até o dia {}.'.format(dfDia2["data_tratada"].max().strftime("%d/%m/%Y")))

#plt.text(0,-150,'Criado por:\nEduardo Braz Rabello')
dados_mundo_corr = dados_brasil[ ['Country/Region', 'dias_corridos', 'Confirmed'] ].copy()

for c in dfDia2['Country/Region'].unique():

    dataset = dados_mundo[dados_mundo['Country/Region'] == c ][ ['dias_corridos', 'Confirmed'] ]

    dados_mundo_corr = pd.merge(dados_mundo_corr, dataset, on='dias_corridos', how='outer', suffixes=(None, f'_{c}'))



dados_mundo_corr.drop(['Country/Region', 'dias_corridos'], axis=1, inplace = True)
teste = dfDia2[dfDia2['Country/Region'] == 'Italy' ][ ['dias_corridos', 'Confirmed'] ]

teste.head()
lista_corr_BR = dados_mundo_corr.corr()['Confirmed'].sort_values(ascending=False).dropna()

lista_corr_BR[lista_corr_BR > 0.95]
dfDia2[ dfDia2['Country/Region'] == 'Portugal']