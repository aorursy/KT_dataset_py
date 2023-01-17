# Importantando as bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as pyo
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objs import *
import plotly.express as px
import folium # biblioteca de plots geográficos
import json as js
from folium.plugins import MarkerCluster, TimestampedGeoJson

%matplotlib inline
covid_19 = pd.read_csv("/kaggle/input/corona-virus-brazil/brazil_covid19.csv") # Importando  dataframe
covid_19.info() # Verificando as informações contidas no dataframe
covid_19.head(10) # Analisando o cabeçalho dos dados
covid_19.columns # Verificando se realmente possui as quatro colunas
covid_19.isna().sum() # Verificando a quantidade de valores NaN por coluna
del covid_19['region'] # Deletei a coluna Region pois não iremos usufruir de seus dados no momento
covid_19.head() # Verificando se realmente foi deletada a coluna no dataframe
covid_19.rename(columns={"date":"data","cases":"casos_confirmados","deaths":"mortes_confirmadas","state":"estado"},inplace=True) # Renomeando colunas, lembre-se de colocar o inplace True
covid_19.head()
df1 = covid_19.groupby('estado').last() # Realiza um groupby em relação aos estados
del df1['data'] # Deletamos a coluna data, pois não iremos utiliza-la nessa análise gráfica
df1.head(27)
df1.columns # Verificamos as novas colunas do dataframe
df1.style.background_gradient(cmap="Reds")
data = [
    go.Bar(x = df1.index, y=df1.casos_confirmados, name="Casos Confirmados"), # Plot do gráfico na vertical de Suspeitos X Estado
    go.Bar(x = df1.index, y=df1.mortes_confirmadas, name="Mortes Confirmadas"),  # Plot do gráfico na vertical Casos X Estado
]

layout = go.Layout(title="Gráfico do Covid-19 por Estado Brasileiro") # Configuração do Título do Gráfico
fig = go.Figure(data = data, layout = layout)
fig.show()
brazil_map = folium.Map(location=[-15.776250,-47.796619],tiles = "CartoDB positron", zoom_start=5)
#Criando a coluna destinada a latitude e longitude de cada estado

N_Longitude  = [
-70.30146,
-36.372506,
-50.79137,
-65.856064,
-38.343577,
-38.647867,
-47.796619,
-40.308865,
-49.836224,
-45.338388,
-56.921101,
-54.545921,
-44.555031,
-52.021542,
-36.781952,
-52.319211,
-36.954105,
-41.736591,
-36.19577,
-51.180061,
-43.40579,
-61.319698,
-61.856049,
-51.092289,
-37.385658,
-46.633308,
-48.33466
]

N_Latitude = [
-9.12893,
-9.672838,
2.05189,
-3.416843,
-12.506686,
-3.931844,
-15.77625,
-19.183422,
-15.827037,
-5.703448,
-12.681871,
-20.61817,
-18.512178,
-25.252089,
-7.239961,
-5.125989,
-8.813717,
-6.40271,
-5.41738,
-29.167391,
-22.939106,
-11.83131,
1.84565,
-27.6534,
-10.574093,
-23.55052,
-10.18451
]
df1.insert(loc=2,column='Latitude',value=N_Latitude) # Inserindo a lista Latitude na coluna do df1
df1.insert(loc=3,column='Longitude',value=N_Longitude) # Inserindo a lista Longitude na coluna do df1
df1.head(27)
df3 = df1.reset_index(drop=False)
df3.head()
# Criando interação para criar os marcadores de todos os estado baseados na Latitude e Longitude
tooltip= "Covid-19 Information"

for i, row in df3.iterrows():
    html="""
    <h2>Information</h1>
        State: {}<br>
        Cases: {}<br>
        Deaths: {}<br>
    """.format(row['estado'],row['casos_confirmados'],row['mortes_confirmadas'])
    folium.Marker([row["Latitude"],row["Longitude"]],
                  popup = html,
                  tooltip = tooltip,
                  icon = folium.Icon(color='red', icon='info-sign')).add_to(brazil_map)
brazil_map
from folium.plugins import HeatMap
brazil_heatmap = folium.Map(location=[-15.776250,-47.796619],tiles = "stamentoner", zoom_start=5)
del df3['estado']
df3.head()
# Criando o Heatmap de casos_confirmados 
locais = df3[["Latitude", "Longitude", "casos_confirmados"]].values.tolist()
HeatMap(locais, radius=20).add_to(brazil_heatmap)
brazil_heatmap
# Criando o Heatmap de mortes_confirmadas
locais = df3[["Latitude", "Longitude", "mortes_confirmadas"]].values.tolist()
HeatMap(locais, radius=20).add_to(brazil_heatmap)
brazil_heatmap
df1.head()
df4 = df1.reset_index(drop=False)
df4.head()
del df4['estado']
del df4['Latitude']
del df4['Longitude']
df4.sum()
df4['casos_confirmados']
d = {'casos_confirmados' : df4['casos_confirmados'].sum(), 'mortes_confirmadas': df4['mortes_confirmadas'].sum()}
df5 = pd.DataFrame(d.items(), columns=['Classe','Valor_Total']).set_index('Classe')
df5.head()
bar = sns.barplot(x=df5.index, y="Valor_Total", data=df5)

for p in bar.patches:
    bar.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),ha='center', va='bottom',color= 'black')
    
bar.set_title('Total de Casos Confirmados e Mortes Confirmadas de Covid-19 - Brasil')
covid_19.head()
df6 = covid_19.drop(columns=['estado'])
df6.head()
df7 = df6.groupby(['data'])['casos_confirmados','mortes_confirmadas'].agg('sum')

df7.head()
data = [
    go.Bar(x = df7.index, y=df7.casos_confirmados, name="Casos Confirmados"), # Plot do gráfico na vertical de Suspeitos X Estado
    go.Bar(x =  df7.index, y=df7.mortes_confirmadas, name="Mortes Confirmadas"),  # Plot do gráfico na vertical Casos X Estado
]

layout = go.Layout(title="Gráficos desde O Primeiro Contagio") # Configuração do Título do Gráfico
fig = go.Figure(data = data, layout = layout)
fig.show()