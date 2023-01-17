import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

df1 = df.drop(['Province/State', 'Lat', 'Long'], axis = 1)

df1 = df1.loc[df1['6/22/20'] >= 100000]

df1 = df1.set_index('Country/Region').T

df1 = df1.loc[df1['Brazil'] >= 10]

df1
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (9,7)

df1.plot(logy = True, legend = False, title = 'Ascensão logarítmica da COVID-19 nos países com mais de 100 mil casos confirmados')
df2 = df.drop(['Province/State', 'Lat', 'Long'], axis = 1)

df2 = df2.loc[df2['6/22/20'] >= 300000]

df2 = df2.set_index('Country/Region').T

df2 = df2.loc[df2['Brazil'] >= 10]

df2
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (15,10)

df2.plot(logy = True, title = 'Ascensão logarítmica da COVID-19 nos epicentros do contágio', marker = '*')
df_mortes = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

df_mortes_1 = df_mortes.drop(['Province/State', 'Lat', 'Long'], axis = 1)

df_mortes_1 = df_mortes_1.loc[df_mortes_1['6/22/20'] >= 25000]

df_mortes_1 = df_mortes_1.set_index('Country/Region').T

df_mortes_1 = df_mortes_1.loc[df_mortes_1['Brazil'] >= 1]

df_mortes_1 = df_mortes_1.sort_values(by = '6/22/20', ascending = False, axis = 1)

df_mortes_1
plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (11,8)

df_mortes_1.plot(logy = True, title = 'Ascensão logarítmica dos óbitos por COVID-19 nos países com mais de 25 mil mortes')
df_mortes_1.plot(kind = 'box', title = 'Diagrama de caixas com a quantidade de mortes em países com mais de 25 mil mortes')
df_mortes_1.plot(kind = 'area', title = 'Gráfico de área com as mortes em países com mais de 25 mil mortes')
df_mortes_1.plot(kind = 'hist', bins = 30)
df_mortes_2 = df_mortes.drop(['Province/State', 'Lat', 'Long'], axis = 1)

df_mortes_2 = df_mortes_2.set_index('Country/Region').T

df_mortes_2 = df_mortes_2.sort_values(by = '6/22/20', ascending = False, axis = 1)

df_mortes_2
df_mortes_2.plot(kind = 'area', legend = False, title = 'Gráfico de área com as mortes no planeta')