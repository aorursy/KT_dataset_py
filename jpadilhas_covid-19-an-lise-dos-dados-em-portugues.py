import numpy as np

import pandas as pd



import plotly.graph_objects as go

import plotly.express as px

import plotly.io as pio

pio.templates.default = "plotly_dark"

from plotly.subplots import make_subplots



from pathlib import Path
cleaned_data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', parse_dates=['Date'])

cleaned_data.head()
#atualizando os headers para facilitar a identificação.

cleaned_data.rename(columns={

                     'Province/State':'uf',

                     'Country/Region':'pais',

                     'Lat':'latitude',

                     'Long': 'longitude',

                     'Date': 'data', 

                     'Confirmed': 'confirmado',

                     'Deaths':'mortes',

                     'Recovered':'curado'

                    }, inplace=True)



# casos 

cases = ['confirmado', 'mortes', 'curado', 'ativos']



# casos = confirmado - mortes - curado

cleaned_data['ativos'] = cleaned_data['confirmado'] - cleaned_data['mortes'] - cleaned_data['curado']



# atualização do nome "Mainland China" para "China"

cleaned_data['pais'] = cleaned_data['pais'].replace('Mainland China', 'China')



# Limpando os dados  

cleaned_data[['uf']] = cleaned_data[['uf']].fillna('')

cleaned_data[cases] = cleaned_data[cases].fillna(0)



data = cleaned_data

cleaned_data.head()
print(f"Primeiro registro:     {data['data'].min()}")

print(f"Último registro:       {data['data'].max()}")

print(f"Total de dias:         {data['data'].max() - data['data'].min()}")
grouped = data.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()



fig = px.line(grouped, x="data", y="confirmado", title="Total de infectados no Mundo")

fig.update_layout(

    title="Infectados confirmados no Mundo",

    xaxis_title="Data",

    yaxis_title="Infectados")

fig.show()
data['uf'] = data['uf'].fillna('')

temp = data[[col for col in data.columns if col != 'uf']]



latest = temp[temp['data'] == max(temp['data'])].reset_index()

latest_grouped = latest.groupby('pais')['confirmado', 'mortes'].sum().reset_index()



fig = px.bar(latest_grouped.sort_values('confirmado', ascending=False)[:20][::1], 

             y='confirmado', x='pais',height=700, orientation='v')

fig.update_layout(

    title="Total de casos confirmados no Mundo",

    xaxis_title="País",

    yaxis_title="Infectados")



fig.show()
grouped_china = data[data['pais'] == "China"].reset_index()

grouped_china_date = grouped_china.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()



grouped_italy = data[data['pais'] == "Italy"].reset_index()

grouped_italy_date = grouped_italy.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()



grouped_us = data[data['pais'] == "US"].reset_index()

grouped_us_date = grouped_us.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()



grouped_br = data[data['pais'] == "Brazil"].reset_index()

grouped_br_date = grouped_br.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()



grouped_rest = data[~data['pais'].isin(['China', 'Italy', 'US', 'Brazil'])].reset_index()

grouped_rest_date = grouped_rest.groupby('data')['data', 'confirmado', 'mortes'].sum().reset_index()
plot_titles = ['China', 'Italia', 'EUA', 'Brasil', 'Outros Países']



#china

fig = px.line(grouped_china_date, x="data", y="confirmado", 

              title=f"Casos confirmados: {plot_titles[0].upper()}", 

              color_discrete_sequence=['#F61067'],

              height=500

             )

fig.show()
#italia

fig = px.line(grouped_italy_date, x="data", y="confirmado", 

              title=f"Casos confirmados: {plot_titles[1].upper()}", 

              color_discrete_sequence=['#10eae0'],

              height=500

             )

fig.show()
#eua

fig = px.line(grouped_us_date, x="data", y="confirmado", 

              title=f"Casos confirmados: {plot_titles[2].upper()}", 

              color_discrete_sequence=['#ea1093'],

              height=500

             )

fig.show()
#brasil

fig = px.line(grouped_br_date, x="data", y="confirmado", 

              title=f"Casos confirmados: {plot_titles[3].upper()}", 

              color_discrete_sequence=['#e7ea10'],

              height=500

             )

fig.show()
#mundo

fig = px.line(grouped_rest_date, x="data", y="confirmado", 

              title=f"Casos confirmados: {plot_titles[4].upper()}", 

              color_discrete_sequence=['#10ea67'],

              height=500

             )

fig.show()
fig = px.choropleth(latest_grouped, locations="pais", 

                    locationmode='country names', color="confirmado", 

                    hover_name="pais", range_color=[1,2000], 

                    color_continuous_scale="portland", 

                    title='Países com casos Confirmados')

#fig.update(layout_coloraxis_showscale=False)

fig.show()
countries = pd.read_csv("../input/countries/america.csv")

countries.head()



america = list(countries['country'])

america_grouped_latest = latest_grouped[latest_grouped['pais'].isin(america)]



america_grouped_latest.head()
fig = px.choropleth(america_grouped_latest, locations="pais", 

                    locationmode='country names', color="confirmado", 

                    hover_name="pais", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='Mapa de calor dos países infectados da AMÉRICA DO NORTE', scope='north america', height=800)

fig.show()
fig = px.choropleth(america_grouped_latest, locations="pais", 

                    locationmode='country names', color="confirmado", 

                    hover_name="pais", range_color=[1,2000], 

                    color_continuous_scale='portland', 

                    title='Mapa de calor dos países infectados da AMÉRICA DO SUL', scope='south america', height=800)

fig.show()

fig = px.bar(america_grouped_latest.sort_values('confirmado', ascending=False)[:10][::1], 

             x='pais', y='confirmado', height=700,color_discrete_sequence=['#84DCC6'], orientation='v')

fig.update_layout(

    title="Casos confirmados na América",

    xaxis_title="País",

    yaxis_title="Infectados")

fig.show()
countries.rename(columns={'country': 'pais', 'population' : 'populacao'}, inplace=True)

countries_data = pd.merge(america_grouped_latest, countries, on='pais', how='inner')

countries_data['p_confirmado'] = countries_data['confirmado'] / countries_data['populacao'] * 100

countries_data['p_mortes'] = countries_data['mortes'] / countries_data['confirmado'] * 100



countries_data.head()
fig = px.bar(countries_data.sort_values('p_confirmado', ascending=False)[:20][::1], 

             x='pais', y='p_confirmado', height=700,color_discrete_sequence=['#84DCC6'], orientation='v')

fig.update_layout(

    title="Casos confirmados x População por país (América)",

    xaxis_title="País",

    yaxis_title="Infectados em Percentual da População")

fig.show()
fig = px.line(grouped, x="data", y="mortes", color_discrete_sequence = ['red'])

fig.update_layout(

    title="Mortes confirmadas no Mundo", 

    xaxis_title="Data",

    yaxis_title="Mortes")

fig.show()
temp = grouped.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=True)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas no MUNDO (Ecala Logarítmica)",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
data['uf'] = data['uf'].fillna('')

temp = data[[col for col in data.columns if col != 'uf']]



latest = temp[temp['data'] == max(temp['data'])].reset_index()

latest_grouped = latest.groupby('pais')['mortes'].sum().reset_index()



fig = px.bar(latest_grouped.sort_values('mortes', ascending=False)[:20][::1], 

             y='mortes', x='pais',height=700, orientation='v')

fig.update_layout(

    title="Total de Mortes confirmadas no MUNDO POR PAÍS",

    xaxis_title="País",

    yaxis_title="Mortes")



fig.show()
#china

temp = grouped_china_date.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=True)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas na CHINA",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
#italia

temp = grouped_italy_date.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=True)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas na ITÁLIA",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
#eua

temp = grouped_us_date.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=True)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas na EUA",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
#brazil

temp = grouped_br_date.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=False)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas no BRASIL",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
#rest

temp = grouped_rest_date.melt(id_vars="data", value_vars=['confirmado', 'mortes'], var_name='casos', value_name='count')



fig = px.line(temp, x="data", y="count", color='casos', color_discrete_sequence = ['cyan', 'red'],log_y=False)

fig.update_layout(

    title="Casos confirmados x Mortes confirmadas em Outros Países",

    xaxis_title="Data",

    yaxis_title="Total")



fig.show()
fig = px.bar(america_grouped_latest.sort_values('mortes', ascending=False)[:10][::1], 

             x='pais', y='mortes', height=700,color_discrete_sequence=['red'], orientation='v')

fig.update_layout(

    title="Mortes confirmadas na América",

    xaxis_title="País",

    yaxis_title="Mortes")

fig.show()
fig = px.bar(countries_data.sort_values('mortes', ascending=False)[:10][::1], 

             x='pais', y='p_mortes', height=700,color_discrete_sequence=['red'], orientation='v')

fig.update_layout(

    title="Mortes confirmadas x Casos confirmados",

    xaxis_title="País",

    yaxis_title="Mortes em Percentual em relação aos Casos confirmados")

fig.show()