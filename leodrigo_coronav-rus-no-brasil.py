import datetime
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import folium
from folium import plugins
df = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19_macro.csv')
df.head()
df.info()
primeiro_caso = df['date'].iloc[0]
primeiro_obito = df.query('deaths > 0').date.iloc[0]
print("Primeiro caso de coronavírus: {}".format(primeiro_caso))
print("Primeiro óbito por COVID-19: {}".format(primeiro_obito))
numeros_atualizados = df.groupby(['date']).sum().tail(1)
casos_confirmados = numeros_atualizados['cases'][0]
obitos_confirmados = numeros_atualizados['deaths'][0]
tx_perc_mortalidade = obitos_confirmados/casos_confirmados * 100
now = datetime.datetime.now()
print("Última atualização: {}".format(str(now)))
print("Casos confirmados: {}".format(casos_confirmados))
print("Óbitos confirmados: {}". format(obitos_confirmados))
print("Taxa de letalidade: {:.2f}%". format(tx_perc_mortalidade))
fig = go.Figure()
    
fig.add_trace(go.Scatter(x=df['date'], y=df['cases'],name='Casos confirmados', line=dict(color='Blue', width=3)))
fig.add_trace(go.Scatter(x=df['date'], y=df['deaths'], name='Óbitos', line=dict(color='Red', width=3)))
fig.add_trace(go.Scatter(x=df['date'], y=df['recovered'], name='Casos recuperados',line=dict(color='Green', width=3)))
fig.add_trace(go.Scatter(x=df['date'], y=df['monitoring'], name='Em monitoramento',line=dict(color='Orange', width=3)))

fig.update_layout(title='Avanço do novo coronavírus no Brasil',
                   xaxis_title='Data',
                   yaxis_title='Ocorrência (acumulado)',
                   legend=dict(x=0, y=1))
    
fig.show()
df['new_cases'] = df['cases'].diff(1)
df['new_deaths'] = df['deaths'].diff(1)
df['new_recovered'] = df['recovered'].diff(1)
df['new_monitoring'] = df['monitoring'].diff(1)
fig = go.Figure()
    
fig.add_trace(go.Bar(x=df['date'], y=df['new_cases'],name='Novos casos'))
fig.add_trace(go.Bar(x=df['date'], y=df['new_deaths'], name='Novos óbitos'))

fig.update_layout(title='Novos casos e óbitos',
                 yaxis_title='Ocorrência',
                 legend=dict(x=0, y=1))
    
fig.show()
              
fig = go.Figure()

fig.add_trace(go.Bar(x=df['date'], y=df['new_recovered'], name='Novos casos recuperados', marker_color='Green'))
fig.add_trace(go.Bar(x=df['date'], y=df['new_monitoring'], name='Novos casos em monitoramento', marker_color='Orange'))

fig.update_layout(title='Novos casos recuperado e em monitoramento',
                 yaxis_title='Ocorrência',
                 legend=dict(x=0, y=1))
    
fig.show()
df_estado = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')
df_estado.head()
state_new = {'Acre':'AC',
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
df_estado['state_new'] = df_estado['state'].map(state_new)
df_estado['mort%'] = df_estado['deaths']/df_estado['cases'] * 100
df_ac = df_estado.tail(27)
df_ac
fig = go.Figure()

fig.add_trace(go.Pie(labels=df_ac['region'], values=df_ac['cases'], hole=.5))

fig.update_layout(title='Casos de COVID-19 por região',
                 yaxis_title='Ocorrência',
                 xaxis_title='Estados',
                 xaxis={'categoryorder':'total descending'},
                 legend=dict(x=1, y=1))
    
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=df_ac['state_new'], y=df_ac['cases'], marker_color='Blue'))

fig.update_layout(title='Casos de COVID-19 por estado',
                 yaxis_title='Ocorrência',
                 xaxis_title='Estados',
                 xaxis={'categoryorder':'total descending'},
                 legend=dict(x=0, y=1))
    
fig.show()
fig = go.Figure()

fig.add_trace(go.Pie(labels=df_ac['region'], values=df_ac['deaths'], hole=.5))

fig.update_layout(title='Óbitos de COVID-19 por região',
                 yaxis_title='Ocorrência',
                 xaxis_title='Estados',
                 xaxis={'categoryorder':'total descending'},
                 legend=dict(x=1, y=1))
    
fig.show()
fig = go.Figure()

fig.add_trace(go.Bar(x=df_ac['state_new'], y=df_ac['deaths'], marker_color='Red'))

fig.update_layout(title='Óbitos de COVID-19 por estado',
                 yaxis_title='Ocorrência',
                 xaxis_title='Estados',
                 xaxis={'categoryorder':'total descending'},
                 legend=dict(x=0, y=1))
    
fig.show()
df_cities = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19_cities.csv')
df_cities = df_cities.set_index('name')
df_cities.tail()
att = df_cities[df_cities['date'] == max(df_cities['date'])]
df_coord = pd.read_csv('../input/corona-virus-brazil/brazil_cities_coordinates.csv')
df_coord.head()
result = pd.merge(att, df_coord, left_index=True, right_on='city_name')
result = result.drop(columns={'state', 'code', 'state_code', 'city_code', 'capital'})
result.head()
coordenadas = result[['lat', 'long', 'cases']]
baseMap = folium.Map(
            width="100%",
            height='100%',
            location=[-15.788497, -47.879873],
            tiles='OpenStreetMap',
            zoom_start=4
)
for i in range(0, len(result)):
    folium.Circle(
        location = [result.iloc[i]['lat'], result.iloc[i]['long']],
        color = '#00FF6A',
        fill = '#00A1B3',
        tooltip = '<li><bold> CIDADE: ' + str(result.iloc[i]['city_name']) + 
                  '<li><bold> CASOS: ' + str(result.iloc[i]['cases']) +
                  '<li><bold> ÓBITOS: ' + str(result.iloc[i]['deaths']),
        radius = (2.0)
        
    ).add_to(baseMap)
baseMap = baseMap.add_child(plugins.HeatMap(coordenadas))
baseMap
df_sudeste_hist = df_estado.loc[df_estado['region'] == 'Sudeste']
df_sudeste = df_sudeste_hist.groupby(['date']).mean()
df_nordeste_hist = df_estado.loc[df_estado['region'] == 'Nordeste']
df_nordeste = df_nordeste_hist.groupby(['date']).mean()
df_norte_hist = df_estado.loc[df_estado['region'] == 'Norte']
df_norte = df_norte_hist.groupby(['date']).mean()
df_sul_hist = df_estado.loc[df_estado['region'] == 'Sul']
df_sul = df_sul_hist.groupby(['date']).mean()
df_centro_hist = df_estado.loc[df_estado['region'] == 'Centro-Oeste']
df_centro = df_centro_hist.groupby(['date']).mean()
fig = go.Figure()
    
fig.add_trace(go.Scatter(x=df_sudeste.index, y=df_sudeste['mort%'], name='Sudeste', line=dict(color='Blue', width=2)))
fig.add_trace(go.Scatter(x=df_nordeste.index, y=df_nordeste['mort%'], name='Nordeste', line=dict(color='Orange', width=2)))
fig.add_trace(go.Scatter(x=df_norte.index, y=df_norte['mort%'], name='Norte', line=dict(color='Red', width=2)))
fig.add_trace(go.Scatter(x=df_sul.index, y=df_sul['mort%'], name='Sul', line=dict(color='Green', width=2)))
fig.add_trace(go.Scatter(x=df_centro.index, y=df_centro['mort%'], name='Centro Oeste', line=dict(color='Purple', width=2)))

fig.update_layout(title='Taxa de letalidade por região do Brasil',
                   xaxis_title='Data',
                   yaxis_title='Letalidade %',
                   legend=dict(x=0, y=1))
    
fig.show()