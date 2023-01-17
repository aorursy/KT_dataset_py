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
import plotly

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from matplotlib_bar_util import groupby_operation



df = pd.read_csv('/kaggle/input/vendas/tudo.csv')
df.head()
df.isnull().sum()
groupby_operation(

    df,

    groupbycolumn='product',

    operation_column='price_x',

    operation='sum',

    size=20,

    title='Os 20 produtos de maior receita',

    height=7,

    width=10,

    index=None,

)
groupby_operation(

    df,

    groupbycolumn='name',

    operation_column='name',

    operation='count',

    size=20,

    title='Os clientes mais frequentes',

    height=7,

    width=10,

    index=None,

)
groupby_operation(

    df,

    groupbycolumn='name',

    operation_column='quantity',

    operation='sum',

    size=20,

    title='Os clientes que compram mais',

    height=7,

    width=10,

    index=None,

)
groupby_operation(

    df,

    groupbycolumn='name',

    operation_column='price_x',

    operation='sum',

    size=20,

    title='Os clientes que gastam mais',

    height=7,

    width=10,

    index=None,

)
df2 = df.copy()

df2['updated_at'] = df['updated_at'].str[:7]
groupby_operation(

    df2,

    groupbycolumn='updated_at',

    operation_column='price_x',

    operation='sum',

    size=20,

    title='Histórico de vendas por mês',

    height=7,

    width=10,

    index=None,

)
df['updated_at'] = pd.to_datetime(df['updated_at'], utc=True)



# df.updated_at.max()

# df.updated_at.min()

# df.updated_at.mean()
df.index = df.updated_at



df.head()
vendas_por_mes = df.groupby([df.index.year, df.index.month]).price_x.sum()

vendas_por_mes.index = pd.Series(vendas_por_mes.index.values).apply(lambda x:str(x[0])+'-'+str(x[1]))



layout = dict(title = 'Vendas por mês',

              xaxis = dict(title = 'Mês'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=vendas_por_mes.index, y=vendas_por_mes, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
vendas_por_ano = df.groupby(df.index.year).price_x.sum()



layout = dict(title = 'Vendas por ano',

              xaxis = dict(title = 'Ano'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=vendas_por_ano.index, y=vendas_por_ano, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}

receita_por_dia = df.groupby([df.index.weekday]).price_x.sum()



list(map(lambda x: dias[x], receita_por_dia.index))
layout = dict(title = 'Vendas por dia da semana',

              xaxis = dict(title = 'Dia da semana'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=list(map(lambda x: dias[x], receita_por_dia.index)),

                   y=receita_por_dia, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
receita_por_hora = df.groupby([df.index.hour]).price_x.sum()



layout = dict(title = 'Soma das vendas por hora do dia',

              xaxis = dict(title = 'Horas do dia'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=receita_por_hora.index,

                   y=receita_por_hora, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
quantidade_por_hora = df.groupby([df.index.hour]).quantity.sum()



layout = dict(title = 'Quantidade de vendas por hora do dia',

              xaxis = dict(title = 'Horas do dia'),

              yaxis = dict(title = 'Quantidade ($)'),

              )

data = [go.Scatter(x=quantidade_por_hora.index,

                   y=quantidade_por_hora, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)