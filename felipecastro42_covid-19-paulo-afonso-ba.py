# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import folium

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import matplotlib.pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/covid-19-casos-pa/Covid-19_PA - Pgina2.csv")
df_daily = pd.read_csv('../input/covid-19-casos-pa/Covid-19_PA - Pgina1.csv')

df_daily = pd.DataFrame(data = df_daily).dropna()
import plotly.express as px

import plotly.graph_objects as go

import math as m
i = len(df_daily) -1

pd.DataFrame([int(df_daily.iloc[i, 1]), int(df_daily.iloc[i, 1] - (df_daily.iloc[i, 3] + df_daily.iloc[i, 4])),

                int(df_daily.iloc[i, 4]), int(df_daily.iloc[i, 3]), 

                '{:.2f}'.format((df_daily.iloc[i, 1] / 117782)*100) + '%', 

                '{:.2f}'.format((df_daily.iloc[i, 3]/df_daily.iloc[i, 1])*100) + '%'], 

                index =['Total de casos', 'Casos ativos', 'Recuperados', 'Óbitos', 

                'Porcentagem da população contaminada', 'Porcentagem de óbitos'], columns = ['Números'])   
rol_mean = df_daily['Novos casos'].rolling(14, win_type='triang', min_periods=1, center = True).mean()

rol_mean = pd.Series(rol_mean)

lista =[]

for k in range(0,len(rol_mean)):

       lista.append(round(rol_mean.iloc[k], 2))

lista = pd.Series(lista)

#rol_mean = round(rol_mean)

pd.DataFrame(lista.tail().values, index = df_daily['Data'].tail(), columns = ['Casos novos (média móvel)'])
#df.drop([ 'Lat', 'Long'], axis = 1)

pd.DataFrame(df.Casos.values, index = df.Bairro, columns = ['Casos'])
fig = go.Figure()

fig.add_trace(go.Scatter(x=df_daily['Data'], y=df_daily['Casos (total)'], mode='lines+markers', name='Casos',

                        line = dict(color='rgb(228,26,28)')))

fig.add_trace(go.Scatter(x=df_daily['Data'], y=df_daily['Recuperados (total)'], mode='lines+markers', name='Recuperados',

                        line = dict(color='#2CA02C')))

fig.add_trace(go.Scatter(x=df_daily['Data'], y=df_daily['Óbitos (total)'], mode='lines+markers', name='Óbitos',

                        line = dict(color='black')))

fig.update_layout(legend=dict(x=0,y=1.0))

fig.show()
fig = go.Figure(data = [go.Bar(x =df_daily['Data'], y = df_daily['Novos casos'], name = 'Novos casos')], layout_title_text = 'Novos casos (por dia)')

fig.update_traces(marker_color = 'rgb(228,26,28)')

fig.add_trace(go.Scatter(x=df_daily['Data'],y =  rol_mean,name= 'Média Móvel', mode='lines',line = dict(color='blue')))

fig.update_layout(legend=dict(x=0,y=1.0), bargap = .0)

fig.show()
fig = px.pie(df, values = 'Casos', names = 'Bairro', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update(layout_showlegend = False)

fig.show()
mapa = folium.Map(location = [-9.412239,-38.231131], tiles = 'Carto DB positron', zoom_start = 12.4)
for i in range(0, len(df)):

    folium.Circle(

    location = [df.iloc[i]['Lat'], df.iloc[i]['Long']], stroke = False ,color ='#F80000', fill = '#F80000', fill_opacity = .6,

    tooltip = str(df.iloc[i]['Bairro']) + ': ' + str(df.iloc[i]['Casos']) + ' casos', radius = m.sqrt(int(df.iloc[i]['Casos'])/m.pi)*110).add_to(mapa)    

    
mapa