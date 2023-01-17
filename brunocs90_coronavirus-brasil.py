import numpy as np

import pandas as pd

import os

import plotly.offline as py

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from plotly.subplots import make_subplots
data = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv', sep=',')

data['date'] = pd.to_datetime(data['date'])



#corrige a informação

data.loc[(data.date == '2020-03-28') & (data.state == 'Ceará'), 'deaths'] = 4

data.loc[(data.date == '2020-03-28') & (data.state == 'Rio Grande do Sul'), 'deaths'] = 2



#Retira as informações duplicadas do mesmo dia e do mesmo estado

data = data.drop_duplicates(subset={'date','state'}, keep='last')

data.head()
#Datas da Análise

dateFirstCase = min(data['date']).strftime("%d/%m/%Y")

dateLastCase = max(data['date']).strftime("%d/%m/%Y")



print('Análise feita entre os dias ' + dateFirstCase + ' e ' + dateLastCase)
#Lista de Estados com sua respectiva UF

ufs = {'Acre' : 'AC', 'Alagoas' : 'AL', 'Amapá' : 'AP', 'Amazonas': 'AM',

       'Bahia': 'BA', 'Ceará': 'CE', 'Distrito Federal': 'DF', 'Espírito Santo': 'ES',

       'Goiás' : 'GO', 'Maranhão' : 'MA', 'Mato Grosso': 'MT', 'Mato Grosso do Sul': 'MS',

       'Minas Gerais': 'MG', 'Pará': 'PA', 'Paraíba': 'PB', 'Paraná': 'PR', 'Pernambuco': 'PE',

       'Piauí': 'PI', 'Rio de Janeiro': 'RJ', 'Rio Grande do Norte': 'RN', 'Rio Grande do Sul': 'RS',

       'Rondônia': 'RO', 'Roraima': 'RR', 'Santa Catarina': 'SC', 'São Paulo': 'SP','Sergipe': 'SE','Tocantins' : 'TO'}



regiao = {'Acre' : 'NORTE', 'Alagoas' : 'NORDESTE', 'Amapá' : 'NORTE', 'Amazonas': 'NORTE',

       'Bahia': 'NORDESTE', 'Ceará': 'NORDESTE', 'Distrito Federal': 'CENTRO-OESTE', 'Espírito Santo': 'SUDESTE',

       'Goiás' : 'CENTRO-OESTE', 'Maranhão' : 'NORDESTE', 'Mato Grosso': 'CENTRO-OESTE', 'Mato Grosso do Sul': 'CENTRO-OESTE',

       'Minas Gerais': 'SUDESTE', 'Pará': 'NORTE', 'Paraíba': 'NORDESTE', 'Paraná': 'SUL', 'Pernambuco': 'NORDESTE',

       'Piauí': 'NORDESTE', 'Rio de Janeiro': 'SUDESTE', 'Rio Grande do Norte': 'NORDESTE', 'Rio Grande do Sul': 'SUL',

       'Rondônia': 'NORTE', 'Roraima': 'NORTE', 'Santa Catarina': 'SUL', 'São Paulo': 'SUDESTE','Sergipe': 'NORDESTE','Tocantins' : 'NORTE'}

#Data Frame de agrupamento por estado

dataState = data



#https://cmdlinetips.com/2018/01/how-to-add-a-new-column-to-using-a-dictionary-in-pandas-data-frame/

#Cria uma nova coluna uf de acordo com a coluna state

dataState['uf']= data['state'].map(ufs)

dataState['regiao']= data['state'].map(regiao)



#Busca os casos e as mortes

dataState1 = dataState.groupby(['state'])['cases','deaths','uf', 'regiao'].tail(1)



#Pega a quantidade de casos suspeitos e descartados

aggregations = {

    'state' : lambda x: max(x),

    'uf' : lambda x: max(x),

}

dataState2 = dataState.groupby('state').agg(aggregations)



#cols = list(novo.columns.values)

dataState = dataState1.merge(dataState2, left_on='uf', right_on='uf')

dataState = dataState[['cases', 'deaths', 'state', 'uf', 'regiao']]

dataState = dataState.sort_values(by=['deaths'], ascending=False)

dataState[0:27].style.background_gradient(cmap='Reds', subset=['cases','deaths'])
#https://plotly.com/python/bar-charts/



#Dados apenas com os estados que contêm mortes

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

dataRegiao = dataState.groupby(['regiao'])['cases','deaths','regiao'].agg('sum')

dataRegiao
fig = go.Figure(data=[go.Pie(labels=dataRegiao.index, values=dataRegiao['cases'], pull=[0.0], hole=.3)])

fig.update_layout(showlegend=True, height=500, width=800, title_text='Percentual de Casos por Região no Brasil ' + dateFirstCase + ' a ' + dateLastCase,  xaxis_title="Tipos de Caso") 

fig.show()
dataDay = data.groupby(['date'])['cases','deaths'].agg('sum')



dataDay = dataDay

dataDay['new cases'] =  (dataDay['cases'] - dataDay['cases'].shift(1))

dataDay['new cases'] = dataDay['new cases'].fillna(0).astype(np.int64)

dataDay = dataDay.loc['2020-02-25':]

dataDay.head()

fig = go.Figure(data=[

    go.Scatter(x=dataDay.index.strftime("%d/%m"), y=dataDay['new cases'], textfont_size=10, textposition='top center',mode='lines+markers+text', name='New cases', text=dataDay['new cases'])    

])

  

fig.update_layout(title={'text': 'Número de <b>Novos Casos</b> a partir de 25/02/2020', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'}, 

                  #xaxis_title="Período", yaxis_title="Quantidade", 

                  xaxis_tickangle=-45) 

fig.show()
dataDayAcumulated = data.groupby(['date'])['cases','deaths'].agg('sum')

dataDayAcumulated['date'] = pd.to_datetime(dataDayAcumulated.index) 



dataDayAcumulated = dataDayAcumulated.loc['2020-02-25':]

dataDayAcumulated.head()
fig = go.Figure(data=[

    go.Scatter(x=dataDayAcumulated.index.strftime("%d/%m"), y=dataDayAcumulated['cases'], textfont_size=10, textposition='top center', mode='lines+markers+text', name='Cases', text=dataDayAcumulated['cases']),

    go.Scatter(x=dataDayAcumulated.index.strftime("%d/%m"), y=dataDayAcumulated['deaths'], textfont_size=10, textposition='bottom center', mode='lines+markers+text', marker_color='indianred', name='Deaths', text=dataDayAcumulated['deaths'])

])

  

fig.update_layout(title={'text': 'Número de <b>Casos Acumulados e Mortes</b> por Coronavirus no Brasil a partir de 25/02/2020', 'y':0.9, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'},

                  #xaxis_title="Período", yaxis_title="Quantidade", 

                  xaxis_tickangle=-45, 

                  legend=dict( traceorder="normal", x=0, y=1, bgcolor='rgb(229, 236, 246)', bordercolor="Black", borderwidth=2)

                 ) 

fig.show()