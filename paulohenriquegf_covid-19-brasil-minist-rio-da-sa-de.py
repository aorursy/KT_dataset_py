import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Importando potly
import plotly
import plotly.offline as py

# habilita o modo offline
from plotly.offline import plot
from plotly.offline import iplot
plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')
#importando arquivo CSV
filename = "../input/covid-brasil/arquivo_geral.csv"
covid = pd.read_csv(filename, sep=';', parse_dates=['data'])
#consultando o arquivo importado
covid.sort_values(by='casosAcumulados', ascending=False).head()
#consultando datas inicio e fim
covid.data.describe()
#Agrupando os valores por estado e encontrando o valor maximo de cada estado
ae = covid[['data','regiao','casosAcumulados','obitosAcumulados','casosNovos','obitosNovos']].groupby(covid['estado']).max().sort_values(by = 'casosAcumulados', ascending=False).reset_index()
#Agrupando os valores por data e somando os valores de cada data
acumulado_data = covid[['data','casosAcumulados','obitosAcumulados','casosNovos','obitosNovos',]].groupby(covid['data']).sum().sort_values(by = 'casosAcumulados', ascending=False).reset_index()

#Agrupando os valores por região e somando-os
regiao = ae[['casosAcumulados','obitosAcumulados','casosNovos','obitosNovos']].groupby(ae['regiao']).sum().sort_values(by = 'casosAcumulados', ascending=False).reset_index()
#Fazendo o calculo da letalidade
ae['letalidade %'] = np.round(100*ae['obitosAcumulados']/ae['casosAcumulados'],2)
#NOVOS casos e obitos confirmados no Brasil por dia

trace0 = go.Scatter(
                x=acumulado_data.data,
                y=acumulado_data['casosNovos'],
                name='Casos Novos',
                mode='lines+markers',
                
                )
                
trace1 = go.Scatter(
                x=acumulado_data.data,
                y=acumulado_data['obitosNovos'],
                name='Obitos Novos',
                mode='lines+markers',
                
                )
                
                

data = [trace0,trace1]


layout = go.Layout(title='Casos Novos no Brasil',
                  yaxis={'title':'Casos'},
                  xaxis={'title': 'Periodo'},
                  hovermode="x")
    

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#Casos e Obitos confirmados no Brasil por dia

trace0 = go.Line(
                x=acumulado_data.data,
                y=acumulado_data['casosAcumulados'],
                name='Casos Confirmados',
                mode='lines+markers',
                
                )
                
trace1 = go.Line(
                x=acumulado_data.data,
                y=acumulado_data['obitosAcumulados'],
                name='Obitos Confirmados',
                mode='lines+markers',
                
                )
                
                

data = [trace0,trace1]


layout = go.Layout(title='Casos Totais no Brasil',
                  yaxis={'title':'Casos'},
                  xaxis={'title': 'Periodo'},
                  legend=dict(
                      x=0.4,
                      y=1,
                      bgcolor='rgba(255, 255, 255, 0)',
                      bordercolor='rgba(255, 255, 255, 0)'),
                   hovermode="x"
                  )
    

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
#casos e obitos por estado e porcentagem de letalidade

ae[['estado','casosAcumulados', 'obitosAcumulados','letalidade %']].style.hide_index()\
        .format({'letalidade %' : "{:.2f}"})\
        .background_gradient(cmap='BuPu')\

trace0 = go.Bar(
                y=regiao.regiao,
                x=regiao['casosAcumulados'],
                name='Casos Confirmados',
                orientation='h'
               
                              
                )
trace1 = go.Bar(
                y=regiao.regiao,
                x=regiao['obitosAcumulados'],
                name='Obitos Confirmados',
                orientation='h'
               )               
                
                

data = [trace0,trace1]


layout = go.Layout(title='Casos Confirmados no Brasil por Região',
                  yaxis={'title':'Região'},
                  xaxis={'title': 'Casos Confirmados'},
                  legend=dict(
                      x=0.75,
                      y=1,
                      bgcolor='rgba(255, 255, 255, 0)',
                      bordercolor='rgba(255, 255, 255, 0)')
                 
                  )
                        

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
