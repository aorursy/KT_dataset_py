import plotly
import plotly.graph_objs as go
import pandas as pd
import plotly.offline as py
import numpy as np
# Dataset escolhido: BR_eleitorado_2016_municipio

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
# Gráfico de Linha

agrupamento = (df.groupby(['uf']).sum()["total_eleitores"]).reset_index()
plotly.offline.init_notebook_mode(connected=True)
uf = agrupamento['uf']
total = agrupamento['total_eleitores'].round()

trace = go.Scatter(
    x = uf,
    y = total,
)

all_data = [trace]
plotly.offline.iplot(all_data, filename='basic-line')
# Gráfico de Barras

eleitor_sup_79 = (df.groupby(['uf']).sum()["f_sup_79"]).reset_index()

trace = go.Bar(
    x=agrupamento['uf'],
    y=eleitor_sup_79["f_sup_79"],
    marker=dict(
        color='rgb(100,150,250)'
    ),
)

data = [trace]
layout = go.Layout(
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.iplot(fig, filename='grouped-bar')
# Gráfico de Histograma

x = eleitor_sup_79["f_sup_79"]
data = [go.Histogram(x=x)]
plotly.offline.iplot(data, filename='basic histogram')

# Gráfico de Radar

mask1 = (df.uf == 'SP')
mask2 = (df.uf == 'RJ')
x1 = df[mask1]
x2 = df[mask2]
sp = x1[['f_60_69','f_70_79','f_sup_79']].sum()
rj = x2[['f_60_69','f_70_79','f_sup_79']].sum()
idade = ['60 a 69','70 a 79', 'Superior a 79']
data = [go.Scatterpolar(
      r = sp.values,
      theta = idade,
      fill = 'toself',
        name = 'SP'
    ),
    go.Scatterpolar(
      r = rj.values ,
      theta = idade,
      fill = 'toself',
      name = 'RJ'
    )
       ]
layout = go.Layout(
  polar = dict(
    radialaxis = dict(
      visible = True
    )
  ),
  showlegend = True
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename = "radar/multiple")
# Gráfico de caixa

mask1 = (df.uf == 'SP')
mask2 = (df.uf == 'RJ')
x1 = df[mask1].sum()
x2 = df[mask2].sum()

trace1 = go.Box(
    name = 'São Paulo',
    y=x1
)

trace2 = go.Box(
    name = 'Rio de Janeiro',
    y=x2
)

data = [trace1, trace2]
plotly.offline.iplot(data)

# Gráfico de violino

agrupamento = (df.groupby(['uf']).sum()["total_eleitores"]).reset_index()
plotly.offline.init_notebook_mode(connected=True)
total = agrupamento['total_eleitores'].round()

fig = {
    "data": [
        {
            "type": 'violin',
            "x": 'Brasil',
            "y": total,
            "legendgroup": 'Eleitorado',
            "scalegroup": 'Eleitorado',
            "name": 'Eleitorado',
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

# classificação das variáveis.

tipo = [["uf", "Qualitativa Nominal"],["total_eleitores", "Quantitativa Continua"],["f_sup_79", "Quantitativa Continua"],["f_60_69", "Quantitativa Continua"],["f_70_79", "Quantitativa Continua"]]
tipo = pd.DataFrame(tipo, columns=["Variavel", "Classificação"])
tipo