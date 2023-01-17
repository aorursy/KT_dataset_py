# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
REND = 'kaggle'
def le_dados_caso(arquivo, estado):
    df_br = pd.read_csv(arquivo)
    estado = estado.upper()

    # converte a coluna de data para um formato compreensivel
    df_br['date'] = pd.to_datetime(df_br['date'])

    # caso necessario, preenche as colunas de confirmados e mortos com 0 quando nao ha dados
    #df_br.fillna({'confirmed':0, 'deaths':0, 'death_rate':0, 'confirmed_per_100k_inhabitants':0}, inplace=True)

    # filtra pelo estado desejado
    df_estadual = df_br.loc[(df_br.state==estado) & (df_br.place_type=='state')].sort_values(by='order_for_place')
    
    df_estadual.drop(columns=['city', 'place_type', 'is_last', 'order_for_place'], inplace=True)

    # reseta os indices so pra ficar bonitinho
    df_estadual.reset_index(drop=True, inplace=True)
    
    if df_estadual.isnull().values.any():
        print('CUIDADO: o resultado pode conter NA ou NaN')
    
    return df_estadual
def le_dados_obito(arquivo, estado):
    df_br = pd.read_csv(arquivo)
    estado = estado.upper()

    # converte a coluna de data para um formato compreensivel
    df_br['date'] = pd.to_datetime(df_br['date'])

    # filtra pelo estado desejado
    df_estadual = df_br.query("state==@estado").sort_values(by='date')

    # reseta os indices so pra ficar bonitinho
    df_estadual.reset_index(drop=True, inplace=True)
    
    if df_estadual.isnull().values.any():
        print('CUIDADO: o resultado pode conter NA ou NaN')

    return df_estadual
arq_caso = '/kaggle/input/covid19-brasilio/caso.csv'
arq_obito = '/kaggle/input/covid19-brasilio/obito_cartorio.csv' 
estado = 'SP'
df_caso = le_dados_caso(arq_caso, estado)
df_obito = le_dados_obito(arq_obito, estado)

df_caso.tail()
df_obito.tail()
df_obito.columns
date_ini = datetime.date(2020,1,1)
days_discard = 14
date_fin = df_caso['date'].max() - datetime.timedelta(days=days_discard)

fig0 = make_subplots(rows=1, cols=3, subplot_titles=['insuficiência respiratória', 'pneumonia', 'Covid-19'])

fig0.add_trace(
    go.Scatter(x=df_obito['date'], y=df_obito['deaths_respiratory_failure_2019'],
              mode="lines", line=go.scatter.Line(color="blue")),
    row=1, col=1)
fig0.add_trace(
    go.Scatter(x=df_obito['date'], y=df_obito['deaths_respiratory_failure_2020'],
              mode="lines", line=go.scatter.Line(color="red")),
    row=1, col=1)

fig0.add_trace(
    go.Scatter(x=df_obito['date'], y=df_obito['deaths_pneumonia_2019'],
              mode="lines", line=go.scatter.Line(color="blue")),
    row=1, col=2)
fig0.add_trace(
    go.Scatter(x=df_obito['date'], y=df_obito['deaths_pneumonia_2020'],
              mode="lines", line=go.scatter.Line(color="red")),
    row=1, col=2)

fig0.add_trace(
    go.Scatter(x=df_obito['date'], y=df_obito['deaths_covid19'],
              mode="lines", line=go.scatter.Line(color="red")),
    row=1, col=3)

fig0.update_xaxes(range=[date_ini, date_fin], row=1, col=1)
fig0.update_xaxes(range=[date_ini, date_fin], row=1, col=2)
fig0.update_xaxes(range=[date_ini, date_fin], row=1, col=3)

tit0 = 'Óbitos registrados em cartório 2019 vs 2020 (até ' + str(days_discard) + ' dias atrás) - Estado: ' + estado.upper()
fig0.update_layout(title_text=tit0)

fig0.show(renderer=REND)

fig1 = go.Figure()

fig1.add_trace(go.Scatter(x=df_caso['date'], y=df_caso['deaths'], mode='markers', name='sec saude'))
fig1.add_trace(go.Scatter(x=df_obito['date'], y=df_obito['deaths_covid19'], mode='markers', name='cartorio'))

fig1.update_layout(xaxis_range=[datetime.date(2020,3,1), df_caso['date'].max()])

tit1 = 'Óbitos segundo fonte (estado: ' + estado.upper() + ')'
fig1.update_layout(title_text=tit1)

fig1.show(renderer=REND)