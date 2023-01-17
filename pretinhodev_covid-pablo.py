import numpy as np

import pandas as pd

import geopandas as gp

import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')

dfs = pd.read_csv('../input/corona-virus-brazil/brazil_covid19_cities.csv')

dcs = pd.read_csv('../input/corona-virus-brazil/brazil_cities_coordinates.csv')

dhs = pd.read_csv('../input/corona-virus-brazil/brazil_population_2019.csv', error_bad_lines=False)
dhs.info()
df.info()
dn = df.loc[df['region']=='Nordeste'].drop(columns=['region'])
dn.sort_values(by='state',ascending=True)
dr = dn.sort_values(by=['date'],ascending=False)
rn = dr.head(9).drop(columns=['date']).sort_values(by='cases',ascending=False)
#fig = go.Figure(data=[

#    go.Bar(name='SF Zoo', x=animals, y=[20, 14, 23]),

#    go.Bar(name='LA Zoo', x=animals, y=[12, 18, 29])

#])

# Change the bar mode

#fig.update_layout(barmode='group')

#fig.show()

figrn = px.bar(rn, x='state',y='cases',labels={'cases':'CASOS','state':'ESTADOS'},title='COVID-19 RANKING NORDESTE - CASOS')

figrn.show()
figrn = px.bar(rn, x='state', y='deaths',labels={'deaths':'MORTES','state':'ESTADOS'},title='COVID-19 RANKING NORDESTE - MORTES')

figrn.show()
figrn = px.line(dn, x='date',y='cases',color='state',labels={'cases':'CASOS','state':'ESTADOS','date':'TEMPO'},title='CURVA DE EVOLUÇÃO COVID-19 NO NORDESTE')

figrn.show()
dfs.info()

dhs.info()

dcs.info()
ds = dfs.loc[dfs['state']=='SE']

cm = dhs.loc[dhs['state']=='Sergipe']

coo = dcs.loc[dcs['state_code']==28]
rs = ds.sort_values(by=['date','cases'],ascending=False).head()
#fig = go.Figure(data=[

#    go.Bar(name='CASOS', x=rs['name'], y=rs['cases']),

#    go.Bar(name='MORTES', x=rs['name'], y=rs['deaths'])

#])

# Change the bar mode

#fig.update_layout(barmode='group')

#fig.show()

figrs = px.bar(rs, x='name', y='cases',labels={'cases':'CASOS','name':'MUNICÍPIOS'},title='TOP 5 COVID-19 SERGIPE - CASOS')

figrs.show()
dds = ds[['date','name','cases','deaths']].sort_values(by=['date'],ascending=False).head(75).sort_values(by=['name'],ascending=True)

dcm = cm[['city','population']].sort_values(by=['city'],ascending=True)

dcoo = coo[['city_name','lat','long']].sort_values(by=['city_name'],ascending=True)

dcm.columns = ['name','pop']

dcoo.columns = ['name','lat','long']

rst = pd.merge(dds, dcm, how='outer', on='name').sort_values(by=['cases'],ascending=False).head()

mergecm = pd.merge(dds, dcm, how='outer', on='name')

mergecoo = pd.merge(mergecm, dcoo, how='outer', on='name')

mergecoo.info()
figrs = go.Figure(data=[

    go.Bar(name='POPULAÇÃO', x=rst['name'], y=rst['pop']),

    go.Bar(name='CASOS', x=rst['name'], y=rst['cases'])

])

figrs.update_layout(barmode='overlay')

figrs.show()
ms = ds.sort_values(by=['date','deaths'],ascending=False).head()

figrs = px.bar(ms, x='name', y='deaths',labels={'deaths':'MORTES','name':'MUNICÍPIOS'},title='TOP 5 COVID-19 SERGIPE - MORTES')

figrs.show()
figrs = px.line(ds, x='date',y='cases',color='name',labels={'cases':'CASOS','name':'MUNICÍPIOS','date':'TEMPO'},title='CURVA DE EVOLUÇÃO COVID-19 EM SERGIPE')

figrs.show()