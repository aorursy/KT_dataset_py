# leitura do csv
import pandas as pd
url_parlamentar = "https://raw.githubusercontent.com/matheusmota/dataviz2018/master/resources/datasets/cota_parlamentar_sp.csv"
df = pd.read_csv(url_parlamentar, delimiter=';')
df.head(5)
## Import plotly
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
# Gráfico de linha com os gastos por ano
# Retirando 2018 por ter dados incompletos
serie_ano = df.where(df.numano != 2018).groupby(['numano'])['vlrdocumento'].sum()/100
trace = go.Scatter(
    x = serie_ano.keys(),
    y = serie_ano
)

data = [trace]

layout = dict(title = 'Gastos parlamentares por ano',
              xaxis = dict(title = 'Ano'),
              yaxis = dict(title = 'Gastos em reais'),
              )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='basic-line')
df_group = df.where(df.numano != 2018).groupby(['nudeputadoid','sgpartido']).size().to_frame()
serie_partidos = df_group.groupby(['sgpartido']).size().sort_values(ascending=False)
data = [go.Bar(
            x=serie_partidos.keys(),
            y=serie_partidos,
            text=serie_partidos,
            textposition = 'auto'
    )]

layout = dict(title = 'Parlamentares que obtiveram gastos nos últimos 10 anos por partido',
              xaxis = dict(title = 'Partidos'),
              yaxis = dict(title = 'Quantidade parlamentares'),
              )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='basic-bar')
df_group = df.where(df.numano != 2018)
x = np.random.randn(500)
data = [go.Histogram(x=df_group.nummes)]

layout = dict(title = 'Histograma de gastos emitidos por mês',
              xaxis = dict(title = 'Meses'),
              yaxis = dict(title = 'Qtde de Gastos'),
              )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='basic histogram')
gastos = df.where(df.numano != 2018).groupby(['nummes']).size().sort_values(ascending=False)

def meses(serie):
    ret = dict()
    ret['Jan'] = serie[1]
    ret['Fev'] = serie[2]
    ret['Mar'] = serie[3]
    ret['Abr'] = serie[4]
    ret['Mai'] = serie[5]
    ret['Jun'] = serie[6]
    ret['Jul'] = serie[7]
    ret['Ago'] = serie[8]
    ret['Set'] = serie[9]
    ret['Out'] = serie[10]
    ret['Nov'] = serie[11]
    ret['Dez'] = serie[12]
    return ret

gastos = meses(gastos)
data = [
    go.Scatterpolar(
      r = list(gastos.values()),
      theta = list(gastos),
      fill = 'toself',
      name = 'Group A'
    )
]

layout = go.Layout(
  title = 'Identificação dos meses de menores gastos',
  polar = dict(
    radialaxis = dict(
      visible = False
    )
  ),
  showlegend = False
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "radar/multiple")
serie = df.where(df.numano == 2016).groupby(['nummes'])['vlrdocumento'].sum()/100
dados = go.Box(x=serie)
data = [dados]

layout = dict(title = 'Box Plot da distribuição dos valores gastos por mês no ano de 2016',
              xaxis = dict(title = 'Valores gastos')
              )
fig = dict(data=data, layout=layout)
py.iplot(fig)
fig = {
    "data": [{
        "type": 'violin',
        "y": df.where(df.numano != 2018).groupby(['numano'])['vlrdocumento'].sum()/100,
        "box": {
            "visible": True
        },
        "line": {
            "color": 'black'
        },
        "meanline": {
            "visible": True
        },
        "fillcolor": '#8dd3c7',
        "opacity": 0.6,
        "x0": 'Total de gastos'
    }],
    "layout" : {
        "title": "Distribuição por ano",
        "yaxis": {
            "zeroline": False,
        }
    }
}

py.iplot(fig, filename = 'violin/basic', validate = False)