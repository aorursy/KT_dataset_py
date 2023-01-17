# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
from plotly import tools
import plotly.graph_objs as go

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
cwur = pd.read_csv("../input/cwurData.csv")
cwur.head()
#Prepare dataset for cwur
cwur12_kr = cwur.loc[(cwur['year'] == 2012) & (cwur['country']=='South Korea')]
cwur13_kr = cwur.loc[(cwur['year'] == 2013) & (cwur['country']=='South Korea')]
cwur14_kr = cwur.loc[(cwur['year'] == 2014) & (cwur['country']=='South Korea')]
cwur15_kr = cwur.loc[(cwur['year'] == 2015) & (cwur['country']=='South Korea')]
cwur12_kr
cwur13_kr
#sort by national rank for each year
cwur14_kr.sort_values(by="national_rank", ascending=False)
cwur15_kr.sort_values(by="national_rank", ascending=False)
cwur15_kr.head()
# create trace1 
trace1 = go.Bar(
                x = cwur14_kr.institution.head(15),
                y = cwur14_kr.national_rank.head(15),
                name = "2014",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = cwur14_kr.world_rank)
# create trace2 
trace2 = go.Bar(
                x = cwur15_kr.institution.head(15),
                y = cwur15_kr.national_rank.head(15),
                name = "2015",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = cwur15_kr.world_rank
                )
layout = go.Layout(
    title='Top 10 universities in South Korea 2014-2015 by CWUR',
    xaxis=dict(
        tickfont=dict(
            size=11,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='National rank',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    barmode='group',
    bargap=0.2,
    bargroupgap=0.2
)

data = [trace1, trace2]
fig = go.Figure(data = data, layout = layout)
iplot(fig)
#prepare the top 400 Universities in each year
top400_14 = cwur.loc[(cwur['year'] == 2014) & (cwur['world_rank'] < 401)]
top400_15 = cwur.loc[(cwur['year'] == 2015) & (cwur['world_rank'] < 401)]
#count out of top 400 by country in 2014                   
count_by_country14 = top400_14.groupby('country')['world_rank'].count()
count_by_country14.sort_values(na_position='last', inplace=True, ascending=False)

#count out of top 400 by country in 2015                   
count_by_country15 = top400_15.groupby('country')['world_rank'].count()
count_by_country15.sort_values(na_position='last', inplace=True, ascending=False)

#여기까지 시리즈
count_by_country15.head()
count_14 = count_by_country14.head(10)
count_15 = count_by_country15.head(10)

#to list
y_country14 = count_14.index.tolist()
y_country15 = count_15.index.tolist()
x_count14 = count_14.values.tolist()
x_count15 = count_15.values.tolist()
#trace0_2014
trace0 = go.Bar(
                x=x_count14,
                y=y_country14,
                marker=dict(color='rgba(171, 50, 96, 0.6)'),
                name='2014',
                orientation='h',
)

#trace1_2015
trace1 = go.Bar(
                x=x_count15,
                y=y_country15,
                marker=dict(color='rgba(12, 50, 196, 0.6)'),
                name='2015',
                orientation='h',
)

#layout
layout = dict(
                title='The number of Universitity in Top 400 by country',
                yaxis=dict(showticklabels=True,domain=[0, 0.85],autorange='reversed'),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85],autorange='reversed'),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)
#prepare the top 400 Universities in each year
top100_14 = cwur.loc[(cwur['year'] == 2014) & (cwur['world_rank'] < 101)]
top100_15 = cwur.loc[(cwur['year'] == 2015) & (cwur['world_rank'] < 101)]

#count out of top 100 by country in 2014                   
count100_by_country14 = top100_14.groupby('country')['world_rank'].count()
count100_by_country14.sort_values(na_position='last', inplace=True, ascending=False)

#count out of top 100 by country in 2015                   
count100_by_country15 = top100_15.groupby('country')['world_rank'].count()
count100_by_country15.sort_values(na_position='last', inplace=True, ascending=False)

count100_14 = count100_by_country14.head(12)
count100_15 = count100_by_country15.head(12)

#to list
y100_country14 = count100_14.index.tolist()
y100_country15 = count100_15.index.tolist()
x100_count14 = count100_14.values.tolist()
x100_count15 = count100_15.values.tolist()

#trace0_2014
trace0 = go.Bar(
                x=x100_count14,
                y=y100_country14,
                marker=dict(color='rgba(171, 50, 96, 0.6)'),
                name='2014',
                orientation='h',
)

#trace1_2015
trace1 = go.Bar(
                x=x100_count15,
                y=y100_country15,
                marker=dict(color='rgba(12, 50, 196, 0.6)'),
                name='2015',
                orientation='h',
)

#layout
layout = dict(
                title='The number of Universitity in Top 100 by country',
                yaxis=dict(showticklabels=True,domain=[0, 0.85],autorange='reversed'),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85],autorange='reversed'),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)
# prepare data frame
pu = cwur[cwur['institution'] =='Peking University']
snu = cwur[cwur['institution'] =='Seoul National University']
ut = cwur[cwur['institution'] =='University of Tokyo']

# Creating trace1
trace1 = go.Scatter(
                    x = pu.year,
                    y = pu.score,
                    mode = "lines+markers",
                    name = "Peking University(CHN)",
                    marker = dict(color = 'rgb(171, 50, 96)'),
                    text= pu.world_rank)
# Creating trace2
trace2 = go.Scatter(
                    x = snu.year,
                    y = snu.score,
                    mode = "lines+markers",
                    name = "Seoul National University(KOR)",
                    marker = dict(color = 'rgb(50, 96, 171)'),
                    text= snu.world_rank)

trace3 = go.Scatter(
                    x = ut.year,
                    y = ut.score,
                    mode = "lines+markers",
                    name = "University of Tokyo(JPN)",
                    marker = dict(color = 'rgb(50, 171, 96)'),
                    text= ut.world_rank)

data = [trace1, trace2, trace3]
layout = dict(title = 'Top univerisity of S.Korea, Japan and China',
              xaxis= dict(title= 'Year',zeroline= False,dtick=1),
              yaxis= dict(title= 'Score',zeroline= False)
             )

fig = dict(data = data, layout = layout)
iplot(fig)

pu.head()
snu.head()
ut.head()