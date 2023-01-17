import numpy as np
import pandas as pd
import csv
%matplotlib inline
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
dtSentimento = pd.read_csv('../input/Analise_De_Sentimentos.csv', delimiter=',', encoding = 'cp1252', sep=";")
dtSentimento['MARCA_OTICA'] = dtSentimento['MARCA_OTICA'].apply(lambda x: '{0:0>9}'.format(x))
dtSentimento['CODIGO_DO_PRESTADOR'] = dtSentimento['CODIGO_DO_PRESTADOR'].apply(lambda x: '{0:0>8}'.format(x))
dtSentimento.head()
dtSentimento = dtSentimento.drop(['DDD_COB','TELEFONE_COB','DDD_COM','TELEFONE_COM','DDD_RES','TELEFONE_RES','DDD_RESP','TELEFONE_RESP','DDD_CEL','TELEFONE_CEL','EMAIL','NIVEL_DE_REDE','SEGMENTACAO_REDE'], axis=1)
dtSentimento.head(10)
tipo = type(dtSentimento['DATA_HORA_ATENDIMENTO'][0])
tipo
dtSentimento['DATA_HORA_ATENDIMENTO'] = pd.to_datetime(dtSentimento['DATA_HORA_ATENDIMENTO'])
DATA_HORA_ATENDIMENTO = type(dtSentimento['DATA_HORA_ATENDIMENTO'][0])
DATA_HORA_ATENDIMENTO
dtSentimento['YEAR'] = dtSentimento['DATA_HORA_ATENDIMENTO'].dt.year
dtSentimento['MONTH'] = dtSentimento['DATA_HORA_ATENDIMENTO'].dt.month
dtSentimento['DAY'] = dtSentimento['DATA_HORA_ATENDIMENTO'].dt.day
nulos = dtSentimento.isnull().sum(axis = 0)
pd.DataFrame(nulos)
#Valores fora da faixa esperada são os valores de idade = 0
idade0 = dtSentimento.groupby(dtSentimento['IDADE'] == 0).size()
pd.DataFrame(idade0).reset_index()
ano0 = dtSentimento.groupby(dtSentimento['YEAR'] == 0).size()
pd.DataFrame(ano0).reset_index()
#Preencher os campos NaN com 0
dtSentimento = dtSentimento.fillna(0)
pd.DataFrame(dtSentimento.isnull().sum(axis = 0))
segmentacao_prestador = pd.DataFrame(dtSentimento.groupby(['SEGMENTACAO'])['CODIGO_DO_PRESTADOR'].count().reset_index())
segmentacao_prestador
ocorrencias_estado = pd.DataFrame(dtSentimento.groupby(['UF_BENEFICIARIO'])['MARCA_OTICA'].count().sort_values(ascending = False).reset_index())
ocorrencias_estado.head(10)
ocorrencias_nota = pd.DataFrame(dtSentimento.groupby(['NOTA'])['MARCA_OTICA'].count().reset_index())
ocorrencias_nota.head(11)
procedimento_idade = pd.DataFrame(dtSentimento.groupby(['IDADE','ESPECIALIDADE_PROCEDIMENTO']).size().reset_index())
procedimento_idade.head(800)
segmentacao_regiao = pd.DataFrame(dtSentimento.groupby(['REGIONAL','SEGMENTACAO']).size().reset_index())
segmentacao_regiao
nota_mensal = pd.DataFrame(dtSentimento.groupby('MONTH')['NOTA'].mean().reset_index())
nota_mensal
segmentacao = dtSentimento.groupby(['SEGMENTACAO'])['CODIGO_DO_PRESTADOR'].count()

labels = ['Consultório','Hospital','SADT']
values = segmentacao

trace = go.Pie(labels=labels, values=values)

py.iplot([trace], filename='basic_pie_chart')
data = []
trace1 = go.Bar(
    
    x = ocorrencias_estado['UF_BENEFICIARIO'],
    y =  ocorrencias_estado['MARCA_OTICA'],
    text = ocorrencias_estado['UF_BENEFICIARIO'] ,
    name = 'OCORRENCIA ',
    textposition = 'auto',
    marker=dict(
        color='blue',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),   
)

data = [trace1]

plotly.offline.iplot(data, filename='grouped-bar-direct-labels')

trace0 = go.Scatter(
    x = ocorrencias_nota['NOTA'],
    y = ocorrencias_nota['MARCA_OTICA'],
    mode = 'markers',
    name = 'markers'
)
data = [trace0]
py.iplot(data, filename='scatter-mode')

data = []
trace1 = go.Bar(
    
    x = ocorrencias_estado['UF_BENEFICIARIO'] ,
    y =  ocorrencias_estado['MARCA_OTICA'],
    text = ocorrencias_estado['UF_BENEFICIARIO'] ,
    name = 'OCORRENCIA ',
    textposition = 'auto',
    marker=dict(
        color='blue',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),   
)

data = [trace1]

plotly.offline.iplot(data, filename='grouped-bar-direct-labels')
trace0 = go.Scatter(
    x= nota_mensal['MONTH'],
    y= nota_mensal['NOTA']
)

data = [trace0]

plotly.offline.iplot(data)
ocorrencias_nota = pd.DataFrame(dtSentimento.groupby(['NOTA'])['MARCA_OTICA'].count().reset_index())
classificacao = [["NOME_DO_PRESTADOR", "Qualitativa Nominal"],["CODIGO_DO_PRESTADOR","Qualitativa Ordinais"],
                 ["REGIONAL_PRESTADOR","Qualitativa Nominal"],["SEGMENTACAO","Qualitativa Nominal"],
                 ["NUM_PEDIDO_AUTORIZACAO","Qualitativa Ordinais"]
                ]
classificacao = pd.DataFrame(classificacao, columns=["Variavel", "Classificação"])
classificacao
            
     
