import pandas as pd

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
df = pd.read_csv('../input/vendas/tudo.csv')
df.head() 
df.isnull().sum()
df.groupby('product').price_x.sum().sort_values(ascending=False)[:20]
from matplotlib_bar_util import groupby_operation
groupby_operation(df, groupbycolumn='product',

                    operation_column='price_x',

                    operation='sum',

                    title='Produtos de maior receita (20+ frequentes)',

                    size=20,

                    height=8,

                    width=10)
groupby_operation(df, groupbycolumn='name',

                    operation_column='name',

                    operation='count',

                    title='Clientes mais frequentes +20)',

                    size=20,

                    height=8,

                    width=10)
df['name'].value_counts()
groupby_operation(df, groupbycolumn='name',

                    operation_column='quantity',

                    operation='sum',

                    title='Clientes compram mais (+20)',

                    size=20,

                    height=8,

                    width=10)
groupby_operation(df, groupbycolumn='name',

                    operation_column='quantity',

                    operation='sum',

                    title='Clientes compram mais produtos (+20)',

                    size=20,

                    height=8,

                    width=10)
groupby_operation(df, groupbycolumn='name',

                    operation_column='price_x',

                    operation='sum',

                    title='Clientes gastam mais R$ (+20)',

                    size=20,

                    height=8,

                    width=10)
groupby_operation(df, groupbycolumn='updated_at',

                    operation_column='created_at',

                    operation='count',

                    title='Meses que mais compraram',

                    size=12,

                    height=8,

                    width=10)
df['updated_at'] = pd.to_datetime(df.updated_at, utc=True)
df.updated_at.max()
df.updated_at.min()
df.updated_at.mean()
df.head()
df.index = df.updated_at
df.head()
vendas_por_mes = df.groupby([df.index.year, df.index.month]).price_x.sum()
vendas_por_mes
vendas_por_mes.index = pd.Series(vendas_por_mes.index.values).apply(lambda x: str(x[0])+'-'+str(x[1]))
vendas_por_mes
layout = dict(title = 'Vendas por mês',

              xaxis = dict(title = 'Mês'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=vendas_por_mes.index, y=vendas_por_mes, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
vendas_por_ano = df.groupby([df.index.year, df.index.year]).price_x.sum()
vendas_por_ano
vendas_por_ano.index = pd.Series(vendas_por_ano.index.values).apply(lambda x: str(x[0])+'-'+str(x[1]))
vendas_por_ano
layout = dict(title = 'Vendas por ano',

              xaxis = dict(title = 'Ano'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=vendas_por_ano.index, y=vendas_por_ano, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
dias = {0: 'Segunda', 1:'Terça', 2:'Quarta', 3:'Quinta', 4:'Sexta', 5:'Sábado', 6:'Domingo'}
receita_por_dia = df.groupby([df.index.weekday]).price_x.sum()

layout = dict(title = "Vendas por dia da Semana",

             xaxis = dict(title = 'Dia da Semana'),

             yaxis = dict(title = 'Vendas ($)'),

             )

data = [go.Scatter(x=list(map(lambda x: dias[x], receita_por_dia.index)),

                  y=receita_por_dia, mode='lines+markers',

                  marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)