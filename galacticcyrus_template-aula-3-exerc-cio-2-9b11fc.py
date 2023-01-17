from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
dfEleitorado = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
dfEleitorado.dataframeName = 'BR_eleitorado_2016_municipio.csv'
dfEleitorado = dfEleitorado.drop(['cod_municipio_tse', 'gen_nao_informado'], axis=1)
dfEleitorado
#Um gráfico de linha
mask = (dfEleitorado.uf == 'SP')
x = dfEleitorado[mask]
dfeleitoresIdade = x[['f_16', 'f_17', 'f_18_20' ,'f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']].sum()
idade = ['16','17','18 a 20','21 a 24','25 a 34','35 a 44','45 a 59','60 a 69','70 a 79','Superior a 79']
trace = go.Scatter(
                x = idade,
                y = dfeleitoresIdade.values,
                marker = dict(color = 'rgba(84, 92, 229, 0.9)', line=dict(color='rgb(0,0,0)',width=2)),
                text = dfeleitoresIdade.index)
layout = go.Layout(
    title='Total de eleitores por idade em SP)',
    xaxis=dict(
        title='Anos',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Número de eleitores (Em milhões)',
        titlefont=dict(
            size=16
        )
    )
)

py.iplot(go.Figure(data = [trace], layout=layout))
#Um gráfico de barras
dfEleporEstado = dfEleitorado[['uf','total_eleitores']].groupby(by="uf")
trace = go.Bar(
                x = dfEleporEstado.sum().index,
                y = dfEleporEstado.total_eleitores.sum(),
                marker = dict(color = 'rgba(84, 92, 229, 0.9)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                )
layout = go.Layout(
    title='Quantidade de eleitores por UF',
    xaxis=dict(
        title='Estado',
        titlefont=dict(
            size=16
        )
    ),
    yaxis=dict(
        title='Quantidade de eleitores',
        titlefont=dict(
            size=16
        )
    )
) 

py.iplot(go.Figure(data = [trace], layout=layout))

#Um histograma
dfEleitorado.total_eleitores.nlargest(50)
x = dfEleitorado.total_eleitores.nlargest(50)
data = [go.Histogram(x=x, nbinsx = 50)]

layout = go.Layout(
    title='Histograma - Quantidade eleitores das maiores 50 cidades',
    xaxis=dict(
        title='Range de eleitores'
    ),
    yaxis=dict(
        title='Quantidade de cidades'
    ),
    bargap=0.2,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

x = dfEleitorado[['uf','nome_municipio','total_eleitores']].nlargest(5, 'total_eleitores')
x
#Um gráfico radar
mask1 = (dfEleitorado.uf == 'SP')
mask2 = (dfEleitorado.uf == 'BA')
x1 = dfEleitorado[mask1]
x2 = dfEleitorado[mask2]
dfSP = x1[['f_18_20' ,'f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']].sum()
dfBA = x2[['f_18_20' ,'f_21_24','f_25_34','f_35_44','f_45_59','f_60_69','f_70_79','f_sup_79']].sum()
idade = ['18 a 20','21 a 24','25 a 34','35 a 44','45 a 59','60 a 69','70 a 79']
data = [go.Scatterpolar(
      r = dfSP.values,
      theta = idade,
      fill = 'toself',
        name = 'SP'
    ),
    go.Scatterpolar(
      r = dfBA.values ,
      theta = idade,
      fill = 'toself',
      name = 'BA'
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
#Um gráfico de caixa
dfLargest = dfEleitorado.nlargest(50, 'total_eleitores')

trace0 = go.Box(
    y= dfLargest[10:].gen_feminino.nlargest(10),
    name='F',
    marker=dict(
        color='#f441c1'
    )
)
trace1 = go.Box(
    y= dfLargest[10:].gen_masculino.nlargest(10),
    name='M',
    marker=dict(
        color='#426bf4'
    )
)
data = [trace0, trace1]
layout = go.Layout(
    title='Numero de eleitores Femininos e Masculinos nas cidades entre a decima e vigésima maior população eleitoral (10 - 20)',
    yaxis=dict(
        title = 'Genero por estado',
        zeroline=False
    ),
    boxmode='group'
)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig)

#Um gráfico de violino
mask3 = (dfEleitorado.uf == 'MG')
x3 = dfEleitorado[mask]
fig = {
    "data": [
        {
            "type": 'violin',
            "y": x3.head(10).gen_masculino,
            "legendgroup": 'M',
            "scalegroup": 'M',
            "name": 'Masculino',
            "box": {
                "visible": False
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'blue'
            }
        },
        {
            "type": 'violin',
            "y": x3.head(10).gen_feminino,
            "name": 'Feminino',
            "box": {
                "visible": False
            },
            "meanline": {
                "visible": True
            },
            "line": {
                "color": 'pink'
            }
        }
    ],
    "layout" : {
        "yaxis": {
            "zeroline": False,
        },
        "violinmode": "group"
    }
}
py.iplot(fig, filename = 'violin/grouped', validate = False)
