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
df = pd.read_csv('/kaggle/input/vendas/tudo.csv')
df.head(10)
# %% [code]

import matplotlib.pyplot as plt

import seaborn as sns

import locale

locale.setlocale(locale.LC_ALL, '')



def groupby_operation(dataframe, groupbycolumn, operation_column, operation, 

                      size=None, total=None, title=None, height=10, width=5, index=None):

    fig = plt.figure()

    fig.set_size_inches(width, height)



    ax1 = plt.subplot(1,1,1)



    if operation == 'sum':

        serie = dataframe.groupby(groupbycolumn)[operation_column].sum().sort_values(ascending=True).astype(float)

    elif operation == 'count':

        serie = dataframe.groupby(groupbycolumn)[operation_column].count().sort_values(ascending=True).astype(float)

    else:

        serie = dataframe.groupby(groupbycolumn)[operation_column].count().sort_values(ascending=True).astype(float)

        

    if not total:

        total = serie.sum()

        

    corte = ''

        

    if size and len(serie) > size:

        serie = serie.sort_values(ascending=False)

        serie = serie[:size]

        serie = serie.sort_values(ascending=True)

        corte = ' ({} maiores)'.format(size)

    

    if not title:

        if operation_column:

            column = operation_column

        else:

            column = serie.name

        title = "Soma de {} agrupado por {}{}".format(operation_column, groupbycolumn, corte)

   

    ax1.barh(serie.index, serie, align='center', color='c', ecolor='black')

    percentage = serie/total*100

    number_distance = serie.max()*0.005

    

    for i, v in enumerate(serie):

        pct = locale.format_string('%.2f', percentage[i], True)

        v_str = locale.format_string('%.2f', v, True)

        ax1.text(v+number_distance , i-0.2, '{0} ({1}%)'.format(v_str, pct), color='k')

    ax1.set(title=title,

           xlabel='',

           ylabel='')

    sns.despine(left=True, bottom=True)



    plt.show()



def show_value_counts(serie, column_desc=None, grain='Registers', 

                      size=None, total=None, title=None, height=10, width=5, index=None):

    fig = plt.figure()

    fig.set_size_inches(width, height)



    ax1 = plt.subplot(1,1,1)



    serie = serie.value_counts().sort_values(ascending=True)



    if not total:

        total = serie.sum()

    

    corte = ''

    

    if (index):

        serie = serie.rename(index)

    

    if serie.index.dtype != 'object':

        if serie.index.dtype == 'float64':

            serie.index = serie.index.map(int)

        serie.index = serie.index.map(str)

    serie.index = serie.index.map(str)

    

    if size and len(serie) > size:

        serie = serie.sort_values(ascending=False)

        serie = serie[:size]

        serie = serie.sort_values(ascending=True)

        corte = ' ({} mais frequentes)'.format(size)

    

    if not title:

        if column_desc:

            column = column_desc

        else:

            column = serie.name

        title = "Nº de {} por {}{}".format(grain, column, corte)

   

    ax1.barh(serie.index, serie, align='center', color='c', ecolor='black')

    percentage = serie/total*100

    number_distance = serie.max()*0.005

    

    for i, v in enumerate(serie):

        pct = locale.format_string('%.2f', percentage[i], True)

        ax1.text(v+number_distance , i-0.2, '{0:,} ({1}%)'.format(v, pct), color='k')

    ax1.set(title=title,

           xlabel='',

           ylabel='')

    sns.despine(left=True, bottom=True)



    plt.show()
groupby_operation(df, groupbycolumn='product', operation_column='price_x', operation='sum', 

                      size=21, total=None, title=None, height=8, width=4)
df['product'].value_counts()
show_value_counts(df['product'],size=20)
groupby_operation(df, groupbycolumn='name', operation_column='name', operation='count', 

                      size=21, total=None, title=None, height=8, width=4)
groupby_operation(df, groupbycolumn='name', operation_column='quantity', operation='sum', 

                      size=21, total=None, title=None, height=8, width=4)
df['updated_at'] = pd.to_datetime(df.updated_at, utc=True)

df.index = df.updated_at
vendas_por_mes = df.groupby([df['created_at'].index.year, df['created_at'].index.month]).price_x.sum()
vendas_por_mes.index = pd.Series(vendas_por_mes.index.values).apply(lambda x:str(x[0])+'-'+str(x[1]))
import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go
layout = dict(title = 'Vendas por mês',

              xaxis = dict(title = 'Mês'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=vendas_por_mes.index, y=vendas_por_mes, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)
dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
df.groupby([df.index.weekday]).price_x.sum()
receita_por_dia = df.groupby([df.index.weekday]).price_x.sum()

layout = dict(title = 'Vendas por dia da semana',

              xaxis = dict(title = 'Dia da semana'),

              yaxis = dict(title = 'Vendas ($)'),

              )

data = [go.Scatter(x=list(map(lambda x: dias[x], receita_por_dia.index)),

                   y=receita_por_dia, mode='lines+markers',

                 marker=dict(color='blue'))]

fig = dict(data=data, layout=layout)

iplot(fig)