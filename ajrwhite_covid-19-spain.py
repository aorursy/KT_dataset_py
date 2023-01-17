import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

from IPython.core.display import display, HTML

from IPython.display import Image
Image('../input/comunidades-autonomas/Comunidades autonomas.png', width='75%')
df = pd.read_csv('https://covid19.isciii.es/resources/serie_historica_acumulados.csv',

                 encoding = 'ISO-8859-1')

df.drop(range(len(df)-4, len(df)), inplace=True)
regions = {

    'AN': 'Andalucía',

    'AR': 'Aragón',

    'AS': 'Asturias',

    'CN': 'Canarias',

    'CB': 'Cantabria',

    'CE': 'Ceuta',

    'CL': 'Castilla y León',

    'CM': 'Castilla-La Mancha',

    'CT': 'Catalunya',

    'EX': 'Extremadura',

    'GA': 'Galicia',

    'IB': 'Illes Balears',

    'RI': 'La Rioja',

    'MD': 'Madrid',

    'ML': 'Melilla',

    'MC': 'Murcia',

    'NC': 'Navarra',

    'PV': 'País Vasco',

    'VC': 'Valenciana'

}
df['date'] = pd.to_datetime(df.FECHA, format='%d/%m/%Y')

df['cases'] = df.CASOS.fillna(0)

df['deaths'] = df.Fallecidos.fillna(0)

df['region'] = df.CCAA.map(regions)
df.head()
pop_df = pd.read_csv('../input/spanish-regional-population-density/spanish_communities.csv',

                    encoding='utf-8')
pop_df.head()
pop_df['population'] = pop_df['Población'].str.replace('\s', '').astype(int)
pop_df['density'] = pop_df['Densidad'].str.replace(',', '.').astype(float)
pop_df.head()
df2 = df.merge(pop_df[['Nombre', 'population', 'density']],

               left_on='CCAA', right_on='Nombre', how='left')
df2['cases_per_person'] = df2.cases / df2.population

df2['deaths_per_person'] = df2.deaths / df2.population

df2['cases_by_density'] = df2.cases / df2.density

df2['deaths_by_density'] = df2.deaths / df2.density
fig = px.line(df2[df2.cases >= 1000], x='date', y='cases_per_person',

              color='region',

              title='Spain cases per person by region<br>(log scale; beginning with 1000th case)')

fig.update_layout(yaxis_type="log")

fig.show()
fig = px.line(df2[df2.deaths >= 100], x='date', y='deaths_per_person',

              color='region',

              title='Spain deaths per person by region<br>(log scale; beginning with 100th death)')

fig.update_layout(yaxis_type="log")

fig.show()
fig = px.line(df2[df2.cases >= 1000], x='date', y='cases_by_density',

              color='region',

              title='Spain cases by regional density<br>(log scale; beginning with 1000th case)')

fig.update_layout(yaxis_type="log")

fig.show()
fig = px.line(df2[df2.deaths >= 100], x='date', y='deaths_by_density',

              color='region',

              title='Spain deaths by regional density<br>(log scale; beginning with 100th death)')

fig.update_layout(yaxis_type="log")

fig.show()