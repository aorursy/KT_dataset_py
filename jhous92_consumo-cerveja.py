# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py

import plotly.graph_objs as go

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importando os dados e armazenando

df = pd.read_csv('/kaggle/input/beer-consumption-sao-paulo/Consumo_cerveja.csv')
# Verificando o cabeçalho

df.head()
# Verificando informações do dataset

df.info()
# Verificando quantos dados nulos há

df.isnull().sum()
# Apagando os dados nulos

df.dropna(inplace=True)
# Alterando os tipos dos dados

df['Data'] = pd.to_datetime(df['Data'])

df['Temperatura Media (C)'] = df['Temperatura Media (C)'].str.replace(',', '.').astype('float64')

df['Temperatura Minima (C)'] = df['Temperatura Minima (C)'].str.replace(',', '.').astype('float64')

df['Temperatura Maxima (C)'] = df['Temperatura Maxima (C)'].str.replace(',', '.').astype('float64')

df['Precipitacao (mm)'] = df['Precipitacao (mm)'].str.replace(',', '.').astype('float64')

df['Final de Semana'] = df['Final de Semana'].astype('int32')
# Trocando a colula final de semana de valores inteiros para strings

semana = {0:'Dia de semana', 1:'Final de semana'}

df['Final de Semana'] = df['Final de Semana'].apply(lambda x:semana[x])
# Verificando cabeçalho após limpeza a alterações

df.head()
# Verificando correlações entre as colunas

correlacao = df.corr()



plt.figure(figsize=(10, 7))

sns.heatmap(correlacao, annot=True)
# Agrupar por mês

df['Mes'] = df['Data'].dt.strftime('%Y-%m')

df_mes = df.groupby(df['Mes'], as_index=False).mean()

df_mes.head()
# Analisar as descrições do dataset com o mês agrupado

df_mes.describe()
df_mes_acd = df_mes.sort_values(by='Temperatura Minima (C)', ascending=True)

trace = go.Scatter(x= df_mes_acd['Temperatura Media (C)'],

                  y = df_mes_acd['Consumo de cerveja (litros)'])



data = [trace]



layout = go.Layout(title='Consumo de Ceveja relacionado a temperatura',

                   xaxis={'title':'Temperatura Media (C)'},

                   yaxis={'title':'Consumo de cerveja (litros)'})



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
df_comsumo_semana = df.groupby(df['Final de Semana'], as_index=False).mean()

df_comsumo_semana
trace = go.Pie(labels=df_comsumo_semana['Final de Semana'],

               values=df_comsumo_semana['Consumo de cerveja (litros)'])



data = [trace]



layout = go.Layout(title='Consumo em relação ao dia da semana')



fig = go.Figure (data=data, layout=layout)



py.iplot(fig)
trace = go.Scatter(x = df_mes['Mes'],

                   y = df_mes['Consumo de cerveja (litros)'])



data = [trace]



layout = go.Layout(title='Meses com maiores consumos')



fig = go.Figure(data=data, layout=layout)



py.iplot(fig)