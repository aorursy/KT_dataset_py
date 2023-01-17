# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # chiziqli algebra

import pandas as pd # ma'lumotlar b.n ishlash

# visualaition

import matplotlib.pyplot as plt
import plotly as py
from plotly.offline import init_notebook_mode,iplot, plot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# warnings

import warnings
warnings.filterwarnings('ignore')

# wordcloud kutubxonasi

from wordcloud import WordCloud

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
td = pd.read_csv('../input/world-university-rankings/timesData.csv')

td.tail()
# umumiy ma'lumotlar

td.info()
# raqamli ma'lumotlar

td.describe().T
# dataframe ni 100 tagacham bo'lgan qiymatlarini alohida oldik va shuni ustida ishlimiz

df = td.iloc[:100,:]
df.tail()
import plotly.graph_objs as go

# trace 1 ( trace - grafigdagi chiziq )

# yuqorida import qilgan pylot.graph_objs kutubxonasini go sifatida ishlatamiz

trace1 = go.Scatter(
    
    x = df.world_rank,
    y = df.citations,
    mode='lines',
    name = 'Qabul soni',
    marker = dict(color= 'rgba(50,120,210,0.8)'),
    text = df.university_name
    
)

# go.Scatter() parametrlari x,y-bular o'q, mode-bu chiziqni ko'rinishi(type-lines-, lines+markers-*-), name-plotlarning nomi
# marker-nuqta ranglari,line-chiziq ranglari, text-chiziqni ustiga ko'rsatkichni obargan chiqadigan yozuv

trace2 = go.Scatter(
    
    x = df.world_rank,
    y = df.teaching,
    mode = 'lines+markers',
    name = "O'qitish",
    marker = dict(color='rgba(25, 140, 36, 0.8)'),
    line = dict(color = 'rgba(90,58,150,0.8)', width=2),
    text = df.university_name

)

# trace1 da dunyodagi reytingiga qarab qabul qilinish korsatkichini koramiz
# trace2 da dunyodagi reytingiga qarab oqitish korsatkichini koramiz

data = [trace1, trace2] # data o'zgaruvchimizga trace1 va trace2 ni tenglashtirdik 

layout = dict(title = "Dunyodagi universitetlarning reytingi (o'qitilinish va qabul qilinishiga nisbatan)",
              xaxis = dict(title = 'Dunyo reytingi', ticklen = 10, zeroline = True)
             )

fig = dict(data=data, layout=layout)
iplot(fig)
# Scatter plot lab orqali ko'ramiz
import plotly.graph_objs as go
# Har bir yilni alohida ko'rib chiqamiz

df2014 = td[td.year == 2014].iloc[:100,:]
df2015 = td[td.year == 2015].iloc[:100,:]
df2016 = td[td.year == 2016].iloc[:100,:]

trace1 = go.Scatter(
    
    x = df2014.world_rank,
    y = df2014.citations,
    mode='markers',
    name = '2014',
    marker = dict(color= 'rgba(50,120,210,0.8)'),
    text = df.university_name
    
)

trace2 = go.Scatter(
    
    x = df2015.world_rank,
    y = df2015.citations,
    mode='markers',
    name = '2015',
    marker = dict(color= 'rgba(62,12,210,0.8)'),
    text = df.university_name
    
)

trace3 = go.Scatter(
    
    x = df2016.world_rank,
    y = df2016.citations,
    mode='markers',
    name = '2016',
    marker = dict(color= 'rgba(162,95,250,0.8)'),
    text = df.university_name
    
)

data = [trace1, trace2, trace3]
layout = dict(title = "2014-2015-2016 yil ko'rsatkichlari",
               xaxis = dict(title = 'Dunyo reytingi', ticklen = 7, zeroline = True),
               yaxis = dict(title = "Qabul ko'rsatkichi", ticklen = 7, zeroline = True))
                            
fig = dict(data=data, layout=layout)
iplot(fig)
td = pd.read_csv('../input/world-university-rankings/timesData.csv')

# bar grafigi orqali dataframe ni analiz qlamiz
import plotly.graph_objs as go

df2014 = td[td.year==2014].iloc[:3,:]#df2014 ga dataframe ichidagi year bo'limidan 2014 ga teng bolganlarini top 3tasini tenglashtirdik

trace1 = go.Bar(
    
    x = df2014.university_name,
    y = df2014.citations,
    name = 'Qabul soni',
    marker = dict(color= 'rgba(50,120,210,0.8)',
    line = dict(color= 'rgb(0,0,0)', width = 2)),
    text = df2014.country
    )

trace2 = go.Bar(
    
    x = df2014.university_name,
    y = df2014.teaching,
    name = "O'qish saviyasi",
    marker = dict(color= 'rgba(250,255,210,0.8)',
    line = dict(color= 'rgb(0,0,0)', width = 2)),
    text = df2014.country
    )

data = [trace1, trace2]
layout = go.Layout(barmode = 'group')
fig = go.Figure(data = data, layout = layout)
iplot(fig)
