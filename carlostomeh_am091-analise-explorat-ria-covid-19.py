# Importanto os pacotes que serão utilizados



import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import geopandas as gpd

import plotly.express as px

import fiona



import warnings

warnings.filterwarnings('ignore')



import plotly.graph_objects as go

import folium

import matplotlib.patches as mpatches

import matplotlib.colors as mcolors



from matplotlib.colors import ListedColormap



from fbprophet import Prophet

import random

from datetime import timedelta

from fbprophet.plot import add_changepoints_to_plot

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Curva de casos no Mundo e ao longo do tempo



# Carregando os dados

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',parse_dates=['Last Update'])

df_confirmed = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")



# Preprocessamento



# Renomeando colunas

df.rename(columns={'ObservationDate':'Date', 'Country/Region':'Country'}, inplace=True)

df_confirmed.rename(columns={'Country/Region':'Country'}, inplace=True)



# Filtrando Dataframe

df_confirmed = df_confirmed[["Province/State","Lat","Long","Country"]]
print("Primeiro vamos plotar a curva de casos pelo tempo que ilustra a evoluçaõ da pandemia no mundo")

# Dataframes auxiliares

confirmed = df.groupby(['Date']).sum()['Confirmed'].reset_index()

deaths = df.groupby(['Date']).sum()['Deaths'].reset_index()

recovered = df.groupby(['Date']).sum()['Recovered'].reset_index()



fig = go.Figure()

fig.add_trace(go.Scatter(x=confirmed['Date'], 

                         y=confirmed['Confirmed'],

                         mode='lines+markers',

                         name='Confirmados',

                         line=dict(color='blue', width=2)

                        ))

fig.add_trace(go.Scatter(x=deaths['Date'], 

                         y=deaths['Deaths'],

                         mode='lines+markers',

                         name='Mortos',

                         line=dict(color='Red', width=2)

                        ))

fig.add_trace(go.Scatter(x=recovered['Date'], 

                         y=recovered['Recovered'],

                         mode='lines+markers',

                         name='Recuperados',

                         line=dict(color='Green', width=2)

                        ))

fig.update_layout(

    title='Evolução Global da Pandemia de Coronavirus',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='Némuro de Casos',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    )

)

fig.show()
print("Agora a próxima dúvida natural ao visualizar este grafico é quais países e quando eles foram mais afetados pela pandemia?")





# Curva de casos 



plt.figure(figsize=(16, 10))

teste_curvas = df.copy()

teste_curvas['Country'].replace({'Mainland China': 'China'}, inplace=True)

teste_curvas = teste_curvas.groupby(['Country','Date']).sum()['Confirmed'].reset_index()

teste_curvas['Date'] = pd.to_datetime(teste_curvas['Date'], format='%m/%d/%Y')



# Filtro Pais -> US

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'US']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



# Filtro Pais -> Italy

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'Italy']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



# Filtro Pais -> China

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'China']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



# Filtro Pais -> Russia

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'Russia']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



# Filtro Pais -> Brazil

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'Brazil']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



# Filtro Pais -> India

temp_pais = teste_curvas[['Date', 'Confirmed']][teste_curvas['Country'] == 'India']

plt.plot(temp_pais['Date'], temp_pais['Confirmed'])



plt.title('Curva de casos confirmados de Covid-19 por País', size=30)

plt.xlabel('Data', size=5)

plt.ylabel('Quantidade de Casos', size=30)

plt.legend(['US', 'Italia', 'China', 'Russia', 'Brasil', 'India'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
print("Agora a curva de óbitos ocorridos em decorrencia do contagio com da covid-19...")



# CURVA DISTINTA POR PAIS, pode ser pros 10 maiores



plt.figure(figsize=(16, 10))

teste_curvas = df.copy()

teste_curvas['Country'].replace({'Mainland China': 'China'}, inplace=True)

teste_curvas = teste_curvas.groupby(['Country','Date']).sum()['Deaths'].reset_index()

teste_curvas['Date'] = pd.to_datetime(teste_curvas['Date'], format='%m/%d/%Y')



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'US']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'Italy']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'China']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'Russia']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'Brazil']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



# Filtro Pais

temp_pais = teste_curvas[['Date', 'Deaths']][teste_curvas['Country'] == 'India']

plt.plot(temp_pais['Date'], temp_pais['Deaths'])



plt.title('Curva de mortes confirmadas de óbitos causados pela Covid-19 por País', size=30)

plt.xlabel('Data', size=30)

plt.ylabel('Quantidade de Casos', size=30)

plt.legend(['US', 'Italia', 'China', 'Russia', 'Brasil', 'India'], prop={'size': 20})

plt.xticks(size=20)

plt.yticks(size=20)

plt.show()
print("Neste visão alternativa é possivel observar a disseminação do virus apelo tempo, alguns países tiveram os primeiros casos da doença tardiamente, mas isso não é determinante para a quantidade de casos que serão registrados naquele país.")

# Dataframe com latitue e longitute

df_temp = df.copy()

df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)



# Confirmados

df_temp = df_temp.groupby(['Country','Date']).sum()['Confirmed'].reset_index()

df_confirmed = df_confirmed.drop_duplicates(subset=['Country'])

df_temp_estado = pd.merge(df_temp, df_confirmed[["Lat","Long","Country"]], on=["Country"])

df_temp_estado['Lat'][df_temp_estado.Country == 'France'] = 46.1390

df_temp_estado['Long'][df_temp_estado.Country == 'France'] = -2.4351



fig = px.density_mapbox(df_temp_estado, 

                        z='Confirmed',

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Country", 

                        hover_data=["Confirmed"],   # ,"Deaths","Recovered"], 

                        animation_frame="Date",

                        #color_continuous_scale="Portland",

                        range_color =[0, 200000],

                        radius=30, 

                        zoom=1.8,height=900)

fig.update_layout(title='Casos Confirmados Mundo Afora',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
# Mapa de Calor de óbitos ao longo do tempo



# Dataframe com latitue e longitute

df_temp = df.copy()

df_temp['Country'].replace({'Mainland China': 'China'}, inplace=True)



# Deaths

df_temp = df_temp.groupby(['Country','Date']).sum()['Deaths'].reset_index()

df_confirmed = df_confirmed.drop_duplicates(subset=['Country'])

df_temp_estado = pd.merge(df_temp, df_confirmed[["Lat","Long","Country"]], on=["Country"])

df_temp_estado['Lat'][df_temp_estado.Country == 'France'] = 46.1390

df_temp_estado['Long'][df_temp_estado.Country == 'France'] = -2.4351



fig = px.density_mapbox(df_temp_estado, 

                        z='Deaths',

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Country", 

                        hover_data=["Deaths"],   # ,"Deaths","Recovered"], 

                        animation_frame="Date",

                        #color_continuous_scale="Portland",

                        range_color =[0,20000],

                        radius=30, 

                        zoom=1.8,height=900)

fig.update_layout(title='Óbitos Confirmados Mundo Afora',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(mapbox_style="open-street-map", mapbox_center_lon=0)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})





fig.show()
# Caratecisticas regionais das cidades

path = '/kaggle/input/am091-dataset/infraestrutura_saude.csv'

infra_saude = pd.read_csv(path, sep =';')



# Ministerio da Saude Casos e Mortes

path_min_saude = '/kaggle/input/am091-dataset/casos_covid_min_saude.csv'

min_saude = pd.read_csv(path_min_saude,sep =';', encoding='utf-8')

min_saude.tail()



# Carregando o mapa

path_mapa = '/kaggle/input/mapas-brasil/bcim_2016_21_11_2018.gpkg'

mapa = gpd.read_file(path_mapa, layer = 'lim_municipio_a')

mapa.rename(columns={'geocodigo':'codigo_municipio_completo'}, inplace=True)

mapa['codigo_municipio_completo'] =mapa['codigo_municipio_completo'].astype(np.float) 

#fiona.listlayers(path_mapa)
aux = infra_saude['populacao'] / 10000

infra_saude['Leitos por 10 mil Habitantes antes da pandemia'] = infra_saude['leitos_uti_fev'] / aux 

infra_saude['Leitos por 10 mil Habitantes depois da pandemia'] = infra_saude['leitos_uti_jun'] / aux 



# Aderencia recomendação da OMS antes da pandemia

array = []



for i in range(0,len(infra_saude)):

    if infra_saude['Leitos por 10 mil Habitantes antes da pandemia'][i]>= 1 : a = 'Sim'

    else: a = 'Nao'

    array.append(a)



infra_saude['Atende recomendação da OMS para Leitos de Uti'] = array



# Aderencia recomendação da OMS durante a pandemia

array = []



for i in range(0,len(infra_saude)):

    if infra_saude['Leitos por 10 mil Habitantes depois da pandemia'][i]>= 1 : a = 'Sim'

    else: a = 'Nao'

    array.append(a)



infra_saude['Atende recomendação da OMS para Leitos de Uti depois da pandemia?'] = array



# Adiciona os objetos do mapa

mapa_leito = pd.merge(mapa, infra_saude, on = ['codigo_municipio_completo'])
# Mapa atendia oms antes da pandemia



fig, ax = plt.subplots(figsize=(10, 10))



ax.set_title('Municipios aderentes à recomendação da OMS antes da pandemia', 

             pad = 20, 

             fontdict={'fontsize':20})



red_patch = mpatches.Patch(color='purple', label='Não atende à recomendação da OMS')

yellow_path = mpatches.Patch(color='yellow', label='Atende à recomendação da OMS')





plt.xlabel('Latitude')

plt.ylabel('Longitude')



plt.legend(handles=[red_patch,yellow_path ])





mapa_leito.plot(column= 'Atende recomendação da OMS para Leitos de Uti', cmap='viridis', ax=ax)

plt.show()





labels = ["Sim","Nao"]

values = [len(infra_saude['Atende recomendação da OMS para Leitos de Uti'][infra_saude['Atende recomendação da OMS para Leitos de Uti']=='Sim']), len(infra_saude['Atende recomendação da OMS para Leitos de Uti'][infra_saude['Atende recomendação da OMS para Leitos de Uti']=='Nao'])]





fig = px.pie(infra_saude, values=values, names=labels, color_discrete_sequence=['darkblue','green'], hole=0.5)

fig.update_layout(

    title='Porcentagem de Municipios que atendem à recomendação durante a pandemia',

)

fig.show()
# Grafico de mapa atendia oms depois da pandemia



fig, ax = plt.subplots(figsize=(10, 10))



ax.set_title('Municipios aderentes à recomendação da OMS durante a pandemia', 

             pad = 20, 

             fontdict={'fontsize':20})



red_patch = mpatches.Patch(color='purple', label='Não atende à recomendação da OMS')

yellow_path = mpatches.Patch(color='yellow', label='Atende à recomendação da OMS')

plt.xlabel('Latitude')

plt.ylabel('Longitude')

plt.legend(handles=[red_patch,yellow_path ])





mapa_leito.plot(column= 'Atende recomendação da OMS para Leitos de Uti depois da pandemia?', cmap='viridis', ax=ax)

plt.show()





labels = ["Sim","Nao"]

values = [len(infra_saude['Atende recomendação da OMS para Leitos de Uti depois da pandemia?'][infra_saude['Atende recomendação da OMS para Leitos de Uti depois da pandemia?']=='Sim']), len(infra_saude['Atende recomendação da OMS para Leitos de Uti depois da pandemia?'][infra_saude['Atende recomendação da OMS para Leitos de Uti depois da pandemia?']=='Nao'])]





fig = px.pie(infra_saude, values=values, names=labels, color_discrete_sequence=['darkred','green'], hole=0.5)

fig.update_layout(

    title='Porcentagem de Municipios que atendem à recomendação durante a pandemia',

)

fig.show()
# 20 microrregiões com mais leitos uti geral

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()['leitos_uti_jun'].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp = temp[['leitos_uti_jun', 'nome']].sort_values(by = ['leitos_uti_jun'],ascending = False)[:20].reset_index(drop=True)





fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['leitos_uti_jun'][0:20],

            text=temp['leitos_uti_jun'][0:20],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='20 maiores Microrregiões em Leitos de UTI em Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Leitos de UTI",

        template='plotly_white'



)

fig.show()
# 20 microrregiões com mais leitos uti covid

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()['leitos_covid_junho'].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp = temp[['leitos_covid_junho', 'nome']].sort_values(by = ['leitos_covid_junho'],ascending = False)[:20].reset_index(drop=True)





fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['leitos_covid_junho'][0:20],

            text=temp['leitos_covid_junho'][0:20],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='20 maiores Microrregiões em Leitos de UTI especificos para Covid-19 em Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Leitos de UTI espec. Covid-19",

        template='plotly_white'



)

fig.show()
# 20 microrregiões com mais respiradores

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()['respiradores_junho'].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp = temp[['respiradores_junho', 'nome']].sort_values(by = ['respiradores_junho'],ascending = False)[:20].reset_index(drop=True)





fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['respiradores_junho'][0:20],

            text=temp['respiradores_junho'][0:20],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='20 maiores Microrregiões em Respiradores em Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Respiradores/Ventiladores",

        template='plotly_white'



)

fig.show()
# 30 microrregiões que mais compraram Leitos de UTI a cada 10.000 habitantes

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()[['leitos_uti_fev','leitos_uti_jun', 'populacao']].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp['variação_leitos'] = temp['leitos_uti_jun'] - temp['leitos_uti_fev']

temp['variação_leitos_normalizado'] = temp['variação_leitos']/ temp['populacao']*10000

temp['variação_leitos_normalizado'] = temp['variação_leitos_normalizado'].round(2)



temp = temp[['variação_leitos_normalizado', 'nome']].sort_values(by = ['variação_leitos_normalizado'],ascending = False)[:-10].reset_index(drop=True)



fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['variação_leitos_normalizado'][0:30],

            text=temp['variação_leitos_normalizado'][0:30],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='30 Microrregiões que mais adquiriram leitos de UTI a cada 10.000 habitantes entre Fevereiro e Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Leitos de UTI / 10.000 habitantes",

        template='plotly_white'



)

fig.show()
# 30 microrregiões que mais compraram Leitos de UTI a cada 10.000 habitantes

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()[['leitos_covid_junho', 'populacao']].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp['variação_leitos_normalizado'] = temp['leitos_covid_junho']/ temp['populacao']*10000

temp['variação_leitos_normalizado'] = temp['variação_leitos_normalizado'].round(2)



temp = temp[['variação_leitos_normalizado', 'nome']].sort_values(by = ['variação_leitos_normalizado'],ascending = False)[:-10].reset_index(drop=True)



fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['variação_leitos_normalizado'][0:30],

            text=temp['variação_leitos_normalizado'][0:30],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='30 Microrregiões que mais adquiriram leitos de UTI especificos para Covid-19 a cada 10.000 habitantes entre Fevereiro e Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Leitos de UTI espec. Covid-19 / 10.000 habitantes",

        template='plotly_white'



)

fig.show()
# 30 microrregiões que mais compraram Leitos de UTI a cada 10.000 habitantes

temp = infra_saude.groupby(['cod_microregiao', 'nome_microrregiao', 'nome_uf']).sum()[['respiradores_fev', 'respiradores_junho', 'populacao']].reset_index()



temp['nome'] = temp['nome_microrregiao'] +' - '+temp['nome_uf']

temp['variação_leitos'] = temp['respiradores_junho'] - temp['respiradores_fev']

temp['variação_leitos_normalizado'] = temp['variação_leitos']/ temp['populacao']*10000

temp['variação_leitos_normalizado'] = temp['variação_leitos_normalizado'].round(2)



temp = temp[['variação_leitos_normalizado', 'nome']].sort_values(by = ['variação_leitos_normalizado'],ascending = False)[:-10].reset_index(drop=True)



fig = go.Figure(data=[go.Bar(

            x=temp['nome'][0:20], y=temp['variação_leitos_normalizado'][0:30],

            text=temp['variação_leitos_normalizado'][0:30],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='30 Microrregiões que mais adquiriram Respiradores a cada 10.000 habitantes entre Fevereiro e Junho de 2020',

    xaxis_title="Microregiões",

    yaxis_title="Respiradores / 10.000 habitantes",

        template='plotly_white'



)

fig.show()
# Grafico de barra -  municipio com maior incidencia ?



filtro_data = min_saude[['data', 'casosAcumulado', 'obitosAcumulado', 'codmun', 'municipio','estado']][(min_saude['data'] == '15/08/2020') & (min_saude['codmun'].isnull() != True)].copy()

filtro_data['codmun'] = filtro_data['codmun'].astype(np.int64)

filtro_data.rename(columns={'codmun':'codigo_municipio_sem_digito'}, inplace= True)



final_df = pd.merge(filtro_data, infra_saude[["populacao","codigo_municipio_sem_digito"]], on=["codigo_municipio_sem_digito"])

final_df['incidencia_caso'] = final_df['casosAcumulado'] / final_df['populacao'] * 10000

final_df['incidencia_mortes'] = final_df['obitosAcumulado'] / final_df['populacao'] * 10000



final_df['incidencia_caso'], final_df['incidencia_mortes'] = final_df['incidencia_caso'].round(1), final_df['incidencia_mortes'].round(1)



# DataTable Gradient

queste = final_df[['municipio','populacao','casosAcumulado', 'obitosAcumulado', 'incidencia_caso', 'incidencia_mortes']].copy()

queste = queste.sort_values(by=['populacao'], ascending = False)[0:30].reset_index(drop=True)

queste.style.background_gradient(cmap='Greens')

final_df['municipio'] = final_df['municipio'] +' - '+final_df['estado']

temp = final_df[['incidencia_caso', 'municipio']].sort_values(by = ['incidencia_caso'],ascending = False)[:30].reset_index(drop=True)





fig = go.Figure(data=[go.Bar(

            x=temp['municipio'][0:30], y=temp['incidencia_caso'][0:30],

            text=temp['incidencia_caso'][0:30],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='30 maiores municipios em incidência de casos confirmados por 10.000 habitantes',

    xaxis_title="Municipios",

    yaxis_title="Casos Confirmados",

        template='plotly_white'



)

fig.show()

temp = final_df[['incidencia_mortes', 'municipio']].sort_values(by = ['incidencia_mortes'],ascending = False)[:30].reset_index(drop=True)



fig = go.Figure(data=[go.Bar(

            x=temp['municipio'][0:30], y=temp['incidencia_mortes'][0:30],

            text=temp['incidencia_mortes'][0:30],

            textposition='auto',

            marker_color='black',

            



        )])

fig.update_layout(

    title='30 maiores municipios em incidência de óbitos confirmados por 10.000 habitantes',

    xaxis_title="Municipios",

    yaxis_title="Óbitos Confirmados",

        template='plotly_white'



)

fig.show()

def media_movel_7(dados):

    media_movel = []

    lista_dia=[]

    total_dias = len(dados['y']-1)



    for j in range(8,total_dias):



        a = 0

        for i in range(0,7):

            dia = dados.loc[j][0] + timedelta(-i)

            numerico = dados['y'][dados['ds'] == dia].values

            if numerico >0:

                a = a + numerico.min()

        media = a/7



        #print(media)

        #print(dados.loc[j][0])

        media_movel.append(media)

        lista_dia.append(dados.loc[j][0])

    

    tendencia = pd.DataFrame(columns=['ds', 'y'])

    tendencia['ds'] = lista_dia

    tendencia['y'] = media_movel

    return tendencia;





def df_aux(min_saude, estado):

    corte_estado = min_saude[['data', 'casosNovos']][(min_saude['estado'] == estado) & (min_saude['codmun'].isnull() == True)]



    corte_estado.rename(columns={'casosNovos':'y', 'data':'ds'}, inplace=True)

    corte_estado['ds'] =pd.to_datetime(corte_estado['ds'], format='%d/%m/%Y')

    dados_casos = corte_estado[['ds','y']].copy()





    corte_estado = min_saude[['data', 'obitosNovos']][(min_saude['estado'] == estado) & (min_saude['codmun'].isnull() == True)]

    corte_estado.rename(columns={'obitosNovos':'y', 'data':'ds'}, inplace=True)

    corte_estado['ds'] =pd.to_datetime(corte_estado['ds'], format='%d/%m/%Y')

    dados_mortes = corte_estado[['ds','y']].copy()

    

    return dados_casos.reset_index(drop=True), dados_mortes.reset_index(drop=True);



def plot_casos_mortes(dados_casos,dados_mortes, estado):

    

    plt.figure(figsize=(16, 10))

    plt.bar(dados_casos['ds'], dados_casos['y'])

    media = media_movel_7(dados_casos)

    plt.plot(media['ds'], media['y'], color='orange', linestyle='dashed')

    plt.title('Curva de casos confirmados diarios em '+estado, size=30)

    plt.xlabel('Data', size=30)

    plt.ylabel('# casos confirmados', size=30)

    plt.legend(['Média móvel de {} dias'.format(7), 'de novos casos confirmados diarios'], prop={'size': 20})

    plt.xticks(size=20)

    plt.yticks(size=20)

    plt.show()





    plt.figure(figsize=(16, 10))

    plt.bar(dados_mortes['ds'], dados_mortes['y'])

    media = media_movel_7(dados_mortes)

    plt.plot(media['ds'], media['y'], color='orange', linestyle='dashed')

    plt.title('Curva de novas mortes confirmadas diariamente em '+estado, size=30)

    plt.xlabel('Data', size=30)

    plt.ylabel('# de óbitos', size=30)

    plt.legend(['Média móvel de {} dias'.format(7), 'de novas mortes confirmados diarios'], prop={'size': 20})
teste_casos, teste_mortes = df_aux(min_saude, 'SP')

plot_casos_mortes(teste_casos, teste_mortes, 'São Paulo')
teste_casos, teste_mortes = df_aux(min_saude, 'MG')

plot_casos_mortes(teste_casos, teste_mortes, 'Minas Gerais')
teste_casos, teste_mortes = df_aux(min_saude, 'RJ')

plot_casos_mortes(teste_casos, teste_mortes, 'Rio de Janeiro')
teste_casos, teste_mortes = df_aux(min_saude, 'SC')

plot_casos_mortes(teste_casos, teste_mortes, 'Santa Catarina')
teste_casos, teste_mortes = df_aux(min_saude, 'BA')

plot_casos_mortes(teste_casos, teste_mortes, 'Bahia')
teste_casos, teste_mortes = df_aux(min_saude, 'CE')

plot_casos_mortes(teste_casos, teste_mortes, 'Ceará')
teste_casos, teste_mortes = df_aux(min_saude, 'AM')

plot_casos_mortes(teste_casos, teste_mortes, 'Amazônia')
# Tendencia futura Brasil Casos Confirmados - Média Movel

corte_pais = min_saude[['data', 'casosAcumulado']][min_saude['regiao']== 'Brasil']



corte_pais.rename(columns={'casosAcumulado':'y', 'data':'ds'}, inplace=True)

corte_pais['ds'] =pd.to_datetime(corte_pais['ds'], format='%d/%m/%Y')



temp  = corte_pais[['ds','y']].copy()

dados = media_movel_7(temp)



m = Prophet(interval_width=0.95)

m.fit(dados)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

forecast = m.predict(future_confirmed)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)

m.plot_components(forecast);
# Tendencia futura Brasil novos casos - Média Movel

corte_pais = min_saude[['data', 'casosNovos']][min_saude['regiao']== 'Brasil']



corte_pais.rename(columns={'casosNovos':'y', 'data':'ds'}, inplace=True)

corte_pais['ds'] =pd.to_datetime(corte_pais['ds'], format='%d/%m/%Y')



temp  = corte_pais[['ds','y']].copy()

dados = media_movel_7(temp)





m = Prophet(interval_width=0.95)

m.fit(dados)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

forecast = m.predict(future_confirmed)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)

m.plot_components(forecast);
# Tendencia futura Brasil novos obitos - Média Movel

corte_pais = min_saude[['data', 'obitosNovos']][min_saude['regiao']== 'Brasil']



corte_pais.rename(columns={'obitosNovos':'y', 'data':'ds'}, inplace=True)

corte_pais['ds'] =pd.to_datetime(corte_pais['ds'], format='%d/%m/%Y')



temp  = corte_pais[['ds','y']].copy()

dados = media_movel_7(temp)





m = Prophet(interval_width=0.95)

m.fit(dados)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

forecast = m.predict(future_confirmed)

fig = m.plot(forecast)

a = add_changepoints_to_plot(fig.gca(), m, forecast)

m.plot_components(forecast);
# Tendencia de casos



def plot_prophet_estado(coluna,estado, periodo, string):

    print('Curva predita para ',string, 'para os proximos ', periodo, 'dias')

    

    #filtra o estado escolhido

    corte_estado = min_saude[['data', coluna]][(min_saude['estado'] == estado) & (min_saude['codmun'].isnull() == True)]

    

    corte_estado.rename(columns={coluna:'y', 'data':'ds'}, inplace=True)

    corte_estado['ds'] =pd.to_datetime(corte_estado['ds'], format='%d/%m/%Y')

    

    dados = corte_estado[['ds','y']].copy()

    

    m = Prophet(interval_width=0.95)

    m.fit(dados)

    future = m.make_future_dataframe(periods=periodo)

    future_confirmed = future.copy() # for non-baseline predictions later on

    forecast = m.predict(future_confirmed)

    fig = m.plot(forecast)

    a = add_changepoints_to_plot(fig.gca(), m, forecast)

    m.plot_components(forecast);



    

plot_prophet_estado('casosAcumulado','SP', 30, 'Casos Confirmados em São Paulo')
