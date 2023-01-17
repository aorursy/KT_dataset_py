# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from plotly.subplots import make_subplots
data = pd.read_csv('/kaggle/input/brazil-covid19/corona_brasil.csv', sep=',')

data['date'] = pd.to_datetime(data['date'])

data.head()
#Datas da Análise

dateFirstCase = min(data['date']).strftime("%d/%m/%Y")

dateLastCase = max(data['date']).strftime("%d/%m/%Y")



print('Análise feita entre os dias ' + dateFirstCase + ' e ' + dateLastCase)
regiao = {'AC' : 'NORTE', 'AL' : 'NORDESTE', 'AP' : 'NORTE', 'AM': 'NORTE',

       'BA': 'NORDESTE', 'CE': 'NORDESTE', 'DF': 'CENTRO-OESTE', 'ES': 'SUDESTE',

       'GO' : 'CENTRO-OESTE', 'MA' : 'NORDESTE', 'MT': 'CENTRO-OESTE', 'MS': 'CENTRO-OESTE',

       'MG': 'SUDESTE', 'PA': 'NORTE', 'PB': 'NORDESTE', 'PR': 'SUL', 'PE': 'NORDESTE',

       'PI': 'NORDESTE', 'RJ': 'SUDESTE', 'RN': 'NORDESTE', 'RS': 'SUL',

       'RO': 'NORTE', 'RR': 'NORTE', 'SC': 'SUL', 'SP': 'SUDESTE','SE': 'NORDESTE','TO' : 'NORTE'}
#Data Frame de agrupamento por estado

dataState = data



#https://cmdlinetips.com/2018/01/how-to-add-a-new-column-to-using-a-dictionary-in-pandas-data-frame/

dataState['regiao']= data['uf'].map(regiao)



#Pega a quantidade de casos suspeitos e descartados

aggregations = {

    'cases': lambda x: max(x),

    'deaths' : lambda x: max(x),

    'uf' : lambda x: max(x),

    'regiao': lambda x: max(x)

}

dataState = dataState.groupby('uf').agg(aggregations)

dataState = dataState.sort_values(by=['deaths'], ascending=False)

dataState[0:27].style.background_gradient(cmap='Reds', subset=['cases','deaths'])
dataState = dataState.sort_values(by=['cases'], ascending=False)

dataDeaths = dataState[dataState['deaths'] > 0].sort_values(by='deaths',ascending=False).reset_index(drop=True)



fig = []

fig = make_subplots(rows=2, cols=1, shared_xaxes=False, specs=[[{}],[{}]], subplot_titles=('Cases', 'Deaths'))



fig.add_trace(go.Bar(name='Cases', x=dataState["uf"], y=dataState['cases'],text=dataState['cases'],textposition='auto'), row=1,col=1)

fig.add_trace(go.Bar(name='Deaths', marker_color='indianred', x=dataDeaths["uf"], y=dataDeaths['deaths'],text=dataDeaths['deaths'],textposition='auto'), row=2,col=1)



fig.update_layout(barmode='group', height=700, showlegend=False, title_text='Número de Casos e Morte por Coronavirus entre ' + dateFirstCase + ' a ' + dateLastCase)

fig.show()
#Calcula o total geral do pais

dados = {

'cases': [dataState['cases'].sum()],

'deaths': [dataState['deaths'].sum()],

'lethality rate' : [(dataState['deaths'].sum()/dataState['cases'].sum()) * 100]   

}



dataCountry = pd.DataFrame(dados)

dataCountry
fig = make_subplots(

    rows=1, cols=2,

    specs=[[{"type": "xy"}, {"type": "Indicator"}]],

    #subplot_titles=("Plot 1", "Plot 2")

)



fig.add_trace(go.Bar(name='Cases', x=['Cases'], y=dataCountry['cases'],text=dataCountry['cases'],textposition='auto'),row=1, col=1)

fig.add_trace(go.Bar(name='Deaths', x=['Deaths'], y=dataCountry['deaths'],text=dataCountry['deaths'],textposition='auto'),row=1, col=1)

fig.add_trace(go.Indicator(

    domain = {'x': [0, 1], 'y': [0, 1]},

    value = round(dataCountry['lethality rate'][0],2),

    mode = "gauge+number",

    #title = {'text': "% Taxa de Mortalidade", 'align':'center', 'font':{'size':16}},

    delta = {'reference': 100},

    number = {'font':{'size':40}},

    gauge = {'axis': {'visible': True, 'range': [None, 100], 'nticks':22},

             'shape': 'angular',

             'bar': {'color': 'rgb(99, 110, 250)', 'thickness':1.0},

             'steps' : [{'range': [0, 100], 'color': 'rgb(229, 236, 246)', 'thickness': 1.0}],

             'threshold' : {'line': {'color': 'rgb(239, 85, 59)', 'width': 2}, 'thickness': 1.0, 'value': 100}

            }),

              row=1, col=2)



fig.update_layout(showlegend=False, height=500, 

                  title={'text': 'Total de <b>Casos/Mortes e a Taxa de Letalidade</b> no Brasil entre ' + dateFirstCase + ' a ' + dateLastCase,

                      'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, 

                  xaxis_title="Tipos de Caso",yaxis_title="Quantidade", font={'size':12})
fig = go.Figure(data=[go.Pie(labels=dataDeaths['uf'], values=dataDeaths['deaths'], pull=[0.05], hole=.3)])

fig.update_layout(showlegend=True, height=500, width=800, title_text='Percentual de mortes por Estado ' + dateFirstCase + ' a ' + dateLastCase,  xaxis_title="Tipos de Caso") 

fig.show()
dataRegiao = dataState.groupby(['regiao'])['cases','deaths','regiao'].agg('sum')

dataRegiao
fig = go.Figure(data=[go.Pie(labels=dataRegiao.index, values=dataRegiao['cases'], pull=[0.0], hole=.3)])

fig.update_layout(showlegend=True, height=500, width=800, title_text='Percentual de Casos por Região no Brasil ' + dateFirstCase + ' a ' + dateLastCase,  xaxis_title="Tipos de Caso") 

fig.show()