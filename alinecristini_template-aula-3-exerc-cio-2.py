import pandas as pd
import numpy as np
import plotly
import plotly.graph_objs as go
%matplotlib inline
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
#Leitura do DataSet Cota Parlamentar SP
url ="https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cota_parlamentar_sp.csv"
df = pd.read_csv(url, sep=';')
df.head()
#Verificar valores nulos no DataSet
df.isnull().sum()
#Excluir Coluna que não será utilizada
df.drop(columns= ['txtdescricaoespecificacao'],axis=1, inplace = True)
df.head()
#Classificação das variáveis
variavel = [["dataemissão", "Qualitativa Ordinal"],["nudeputadoid", "Numérica Discreta"],["nulegislatura", "Numérica Discreta"],["numano", "Numérica Discreta"]
            ,["nummes", "Numérica Discreta"],["sgpartido", "Qualitativa Nominal"],["txnomeparlamentar", "Qualitativa Nominal"],["txtdescricao", "Qualitativa Nominal"]
            ,["txtfornecedor", "Qualitativa Nominal"],["vlrdocumento", "Numérica Contínua"]]
variavel = pd.DataFrame(variavel, columns=["Variavel", "Classificação"])
variavel
#Informações do DataSet
df.info()
#Contagem de registros por partido
df['sgpartido'].value_counts()
#Contagem de Tipos de Gastos Cadastrados
df['txtdescricao'].value_counts()
#Agrupador - Valor Total gasto por partido
parlamentares = (df.groupby(['sgpartido']).sum()["vlrdocumento"]).reset_index()

#Gerando o grafico de linhas
plotly.offline.init_notebook_mode(connected=True)

random_x = parlamentares['sgpartido']
random_y = parlamentares['vlrdocumento'].round()

trace1 = go.Scatter(
    x = random_x,
    y = random_y,
)

data = [trace1]
plotly.offline.iplot(data, filename='basic-line')

#Top 10 maiores gastos
df_barras = (df[['txtdescricao','vlrdocumento']].groupby(by=['txtdescricao'])).count().reset_index()
x = df_barras['txtdescricao']
y = df_barras['vlrdocumento'].nlargest(10)

trace1 = go.Bar(
    x= x,
    y=y,
    text=y,
    name='Top 10 gastos dos Parlamentares',
    textposition = 'auto',
    marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5),
        ),
    opacity=0.6
)

data = [trace1]
plotly.offline.iplot(data, filename='grouped-bar-direct-labels')
partidos = (df.groupby(['sgpartido']).sum()["vlrdocumento"]).reset_index()

#Gerando histograma
x = partidos["vlrdocumento"]
data = [go.Histogram(x=x)]

plotly.offline.iplot(data, filename='basic histogram')
#Despesas PT
PT = df[df['sgpartido']=='PT']
PT = PT[PT['numano']==2017].groupby(['nummes']).sum()
PT = PT['vlrdocumento'].reset_index()

#Despesas PSDB
PSDB = df[df['sgpartido']=='PSDB']
PSDB = PSDB[PSDB['numano']==2017].groupby(['nummes']).sum()
PSDB = PSDB['vlrdocumento'].reset_index()

#Despesas PP
PP = df[df['sgpartido']=='PP']
PP = PP[PP['numano']==2017].groupby(['nummes']).sum()
PP = PP['vlrdocumento'].reset_index()

#Despesas DEM
DEM = df[df['sgpartido']=='DEM']
DEM = DEM[DEM['numano']==2017].groupby(['nummes']).sum()
DEM = DEM['vlrdocumento'].reset_index()
#Criando o grafico de radar por mês do ano de 2017 para os partidos que mais tem gastos
data = [
    go.Scatterpolar(
      r = PSDB['vlrdocumento'],
      theta = ["Jan", "Fev", "Abr","Mar", "Mai", "Jun","Jul", "Ago", "Set","Out", "Nov", "Dez"],
      fill = "toself"),
    go.Scatterpolar(
      r = PT['vlrdocumento'],
      theta = ["Jan", "Fev", "Abr","Mar", "Mai", "Jun","Jul", "Ago", "Set","Out", "Nov", "Dez"],
      fill = "toself")
    ]
layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True      
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename = "radar/basic")
#Criando o boxplot para os 4 partidos com mais gastos
trace1 = go.Box(y = PSDB['vlrdocumento'], name = "PSDB")
trace2 = go.Box(y = PT['vlrdocumento'], name = "PT")
trace3 = go.Box(y = PP['vlrdocumento'], name = "PP")
trace4 = go.Box(y = DEM['vlrdocumento'], name = "DEM")

data = [trace1, trace2, trace3, trace4]
plotly.offline.iplot(data)
#Criando o grafico de violino
fig = {
    "data": [
        {
            "type": 'violin',
            "x": df['numano'][df['sgpartido'] == 'PT'],
            "y": df['vlrdocumento'][df['sgpartido'] == 'PT'],
            "legendgroup": 'PT',
            "scalegroup": 'PT',
            "name": 'PT',
            "side": 'negative',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'red'
            }
        },
        {
            "type": 'violin',
            "x": df['numano'][df['sgpartido'] == 'PSDB'],
            "y": df['vlrdocumento'][df['sgpartido'] == 'PSDB'],
            "legendgroup": 'PSDB',
            "scalegroup": 'PSDB',
            "name": 'PSDB',
            "side": 'positive',
            "box": {
                "visible": True
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violingap": 0,
        "violinmode": "overlay"
    }
}


plotly.offline.iplot(fig, filename = 'violin/split', validate = False)
