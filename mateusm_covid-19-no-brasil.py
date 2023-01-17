import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import cm

import plotly as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline

plt.style.use('ggplot')

sns.set_style('whitegrid')

print('Atualizado em:', pd.to_datetime('now'))
df=pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')

df.head()
df.info()
state_codes = {'Acre':'AC',

               'Alagoas':'AL',

               'Amapá':'AP',

               'Amazonas':'AM',

               'Bahia':'BA',

               'Ceará':'CE',

               'Distrito Federal':'DF',

               'Espírito Santo':'ES',

               'Goiás':'GO',

               'Maranhão':'MA',

               'Mato Grosso':'MT',

               'Mato Grosso do Sul':'MS',

               'Minas Gerais': 'MG',

               'Pará':'PA',

               'Paraíba':'PB',

               'Paraná': 'PR',

               'Pernambuco':'PE',

               'Piauí':'PI',

               'Rio de Janeiro':'RJ',

               'Rio Grande do Norte':'RN',

               'Rio Grande do Sul':'RS',

               'Rondônia':'RO',

               'Roraima':'RR',

               'Santa Catarina':'SC',

               'São Paulo':'SP',

               'Sergipe':'SE',

               'Tocantins':'TO',

              }
df['state code'] = df['state'].map(state_codes)
df.head()
n_rows = len(df)

loc_recent = n_rows - 27 # Para decidir o indice inicial da fatia, deve-se diminuir o número total por 27
recent = df[loc_recent:].copy()
recent
recent['cases'].sum()
recent['deaths'].sum()
byregion = recent.groupby(['region']).sum()
byregion[byregion['cases'] == byregion['cases'].max()]['cases']
byregion[byregion['deaths'] == byregion['deaths'].max()]['deaths']
plt.figure(figsize=(12,6))

values = byregion['cases']

labels = byregion.reset_index()['region']

plt.pie(values, labels= values)

plt.title('Número de casos confirmados de covid-19 no Brasil por região')

plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
plt.figure(figsize=(12,6))

values = byregion['cases']

labels = byregion.reset_index()['region']

plt.pie(values, labels=labels, autopct='%1.1f%%',

counterclock=False, pctdistance=0.6, labeldistance=1.2)

plt.title('Distribuição de casos confirmados de covid-19 no Brasil por região')
plt.figure(figsize=(12,6))

values = byregion['deaths']

labels = byregion.reset_index()['region']

plt.pie(values, labels= values)

plt.title('Número de mortes por covid-19 no Brasil por região')

plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
plt.figure(figsize=(12,6))

values = byregion['deaths']

labels = byregion.reset_index()['region']

plt.pie(values, labels=labels, autopct='%1.1f%%',

counterclock=False, pctdistance=0.6, labeldistance=1.2)

plt.title('Distribuição percentual do número de mortes por covid-19 no Brasil por região')
plt.figure(figsize=(12,6))

plt.title('Número de casos confirmados de covid-19 no Brasil por região')

sns.barplot(x='region', y='cases', data=byregion.reset_index(), palette='summer')
plt.figure(figsize=(12,6))

plt.title('Número de mortes por covid-19 no Brasil por região')

sns.barplot(x='region', y='deaths', data=byregion.reset_index(), palette='summer')
df['date'] = pd.to_datetime(df['date'])
bydate = df.groupby(['date']).sum()
bydate.tail()
plt.figure(figsize=(12,6))

bydate['cases'].plot(legend=True)

bydate['deaths'].plot()

plt.title("Gráfico de crescimento do covid-19 no Brasil")

plt.legend(['Casos', 'Mortes'])
sns.lmplot(x='index', y='cases', data=bydate.reset_index().reset_index())
recent[recent['cases'] == recent['cases'].max()]['state']
recent['cases'].max()
recent[recent['deaths'] == recent['deaths'].max()]['state']
recent['deaths'].max()
recent[recent['cases'] == recent['cases'].min()]['state']

recent['cases'].min()
recent[recent['deaths'] == recent['deaths'].min()]['state']
recent['deaths'].min()
plt.figure(figsize=(20,8))

plt.title('Número de casos confirmados de covid-19 no Brasil por estado')

sns.barplot(x='state code', y='cases', data=recent.sort_values(ascending=False, by='cases'), palette='summer')
plt.figure(figsize=(20,8))

plt.title('Número de mortes por covid-19 no Brasil por estado')

sns.barplot(x='state code', y='deaths', data=recent.sort_values(ascending=False, by='deaths'), palette='summer')
recents_northeast = recent[recent['region'] == 'Nordeste'].copy()
recents_northeast
recents_northeast[recents_northeast['cases'] == recents_northeast['cases'].max()]['state']
recents_northeast['cases'].max()
recents_northeast[recents_northeast['cases'] == recents_northeast['cases'].min()]['state']
recents_northeast['cases'].min()
plt.figure(figsize=(12,10))

values = recents_northeast.sort_values(by='cases', ascending=False)['cases']

labels = recents_northeast.sort_values(by='cases',  ascending=False)['state']

explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)

cs=cm.Set1(np.arange(9)/9.)

plt.pie(values, labels= values, explode=explode, colors = cs, autopct='%1.1f%%',

counterclock=False, pctdistance=0.6, labeldistance=1.2)

plt.title('Número de casos confirmados de covid-19 na região nordeste')

plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
plt.figure(figsize=(12,8))

plt.title('Número de casos confirmados de covid-19 no Brasil por estado')

sns.barplot(x='state code', y='cases', data=recents_northeast.sort_values(ascending=False, by='cases'), palette='summer')
plt.figure(figsize=(12,10))

values = recents_northeast.sort_values(by='deaths', ascending=False)['deaths']

labels = recents_northeast.sort_values(by='deaths',  ascending=False)['state']

explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)

cs=cm.Set1(np.arange(9)/9.)

plt.pie(values, labels= values, explode=explode, colors = cs, autopct='%1.1f%%',

counterclock=False, pctdistance=0.6, labeldistance=1.2)

plt.title('Número de mortes por covid-19 na região nordeste')

plt.legend(labels,loc=3, bbox_to_anchor=(1, -0.2, 0.5, 1))
plt.figure(figsize=(12,8))

plt.title('Número de mortes por covid-19 na região nordeste')

sns.barplot(x='state code', y='deaths', data=recents_northeast.sort_values(ascending=False, by='deaths'), palette='summer')