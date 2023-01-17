# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None

import math as math

import matplotlib.pyplot as mtlib

from plotly.offline import plot, iplot, init_notebook_mode

import plotly.graph_objs as go

init_notebook_mode(connected=True)

import plotly.express as px
# Nota: em sistemas regionalizados para o sistema brasileiro o Excel pode salvar em csv delimitando com ";"

file = pd.read_csv("../input/covid19amllet/covid19mallet.csv", sep=',', parse_dates=['Data'])

file.head()
i = len(file.Confirmados) -1

file.loc[[i]]
#i = len(file.Confirmados) -1

print(' Total de casos: ', int(file.iloc[i,4]), 

      '\n Porcentagem da população contaminada: ', '{:.2f}'.format((file.iloc[i, 4] / 13630)*100) + '%',

      '\n Recuperados:', int(file.iloc[i, 6]), 

      '\n Óbitos: ', int(file.iloc[i, 9]),

      '\n Total de casos ativos: ', int(file.iloc[i, 4] - (file.iloc[i, 6] + file.iloc[i, 9]))) 
temp = file.groupby('Data')[['Óbitos','Ativos','Recuperados']].sum().reset_index()

temp = temp.melt(id_vars="Data", value_vars=['Óbitos','Ativos','Recuperados'],

                 var_name='Casos', value_name='Count')
ativos = '#21bf73'

mortes = '#ff2e63'

recuperados = '#fe9801'
fig = px.area(temp, 

              x="Data", 

              y="Count", 

              color='Casos', 

              height=600,

              title='Indice de recuperação dos casos',

              color_discrete_sequence = [ativos,mortes, recuperados])



fig.update_layout(legend=dict(x=0,y=1.0))

fig.update_layout(xaxis_rangeslider_visible=True)



fig.show();
fig = go.Figure()

fig.add_trace(go.Scatter(x=file.Data, y=file.Confirmados, # fill='tonexty' ,

                         name='Total Registrado', fill='tozeroy', mode='lines', line_color='indigo',

                        line = dict(color='#21bf73')))

fig.add_trace(go.Scatter(x=file.Data, y=file.Recuperados,  name='Recuperados', fill='tozeroy',

                        line = dict(color='#fe9801', simplify=True,width=2)))

fig.add_trace(go.Scatter(x=file.Data, y=file.Ativos,  name='Casos Ativos',

                        line = dict(color='#ff2e63', simplify=True,width=2)))# fill down to xaxis

#fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[3, 5, 1, 7], fill='tonexty')) # fill to trace0 y



fig.update_layout(legend=dict(x=0,y=1.0))

fig.update_layout(xaxis_rangeslider_visible=True)



fig.show()

# Média móvel dos ultimos 5 dias

rol_mean = file['Novos Casos'].rolling(7, win_type='triang', min_periods=1, center = True).mean()

rol_mean = pd.DataFrame(data = rol_mean)

#rol_mean = round(rol_mean).astype(int)

rol_mean.tail()
fig = go.Figure(data = [go.Bar(x =file['Data'], y = file['Novos Casos'], name = 'Novos Casos')], layout_title_text = 'Novos casos (por dia)')

fig.update_traces(marker_color = 'rgb(228,26,28)')

fig.add_trace(go.Scatter(x=file['Data'],y =  rol_mean['Novos Casos'],name= 'Média Móvel', mode='lines',line = dict(color='blue')))

fig.update_layout(legend=dict(x=0,y=0.5))

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()