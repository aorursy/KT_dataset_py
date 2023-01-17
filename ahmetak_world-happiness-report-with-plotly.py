
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_2015=pd.read_csv('../input/2015.csv')
df_2016=pd.read_csv('../input/2016.csv')
df_2017=pd.read_csv('../input/2017.csv')
new0_df = ['country','region','happiness_rank','happiness_score','standard_error','economy','family','health','freedom','trust','genorasity','dystopia_residual']
df_2015.columns = new0_df

new1_df = ['country','region','happiness_rank','happiness_score','lower_confidence_interval','upper_confidence_interval','economy','family','health','freedom','trust','genorasity','dystopia_residual']
df_2016.columns = new1_df

new2_df = ['country','happiness_rank','happiness_score','whisker_high','whisker_low','economy','family','health','freedom','genorasity','trust','dystopia_residual']
df_2017.columns = new2_df
df_2016.head(3)
df_2017.head(3)
df_2016.info()

df = df_2016.iloc[:100,:]

trace1 = go.Scatter(
                    x = df.happiness_rank,
                    y = df.economy,
                    mode = "lines",
                    name = "economy",
                    marker = dict(color = 'rgba(106, 11, 27, 0.8)'),
                    text= df.country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.happiness_rank,
                    y = df.health,
                    mode = "lines+markers",
                    name = "Health",
                    marker = dict(color = 'rgba(8, 206, 17, 0.8)'),
                    text= df.country)
trace3 = go.Scatter(
                    x = df.happiness_rank,
                    y = df.freedom,
                    mode = "lines",
                    name = "Freedom",
                    marker = dict(color = 'rgba(6, 11, 270, 0.8)'),
                    text= df.country)
data = [trace1, trace2,trace3]
layout = dict(title = 'Freedom , Health and Economy vs World Ranking of the 100 Most Happy Countries',
              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
df2015 = df_2015.iloc[:100,:]
df2016 = df_2016.iloc[:100,:]
df2017 = df_2017.iloc[:100,:]

trace1 =go.Scatter(
                    x = df2015.happiness_rank,
                    y = df2015.freedom,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2015.country)

trace2 =go.Scatter(
                    x = df2016.happiness_rank,
                    y = df2016.freedom,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.country)

trace3 =go.Scatter(
                    x = df2017.happiness_rank,
                    y = df2017.freedom,
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.country)
data = [trace1, trace2, trace3]
layout = dict(title = 'Freedom rates of the 100 most happy countries in 2015, 2016 and 2017',
              xaxis= dict(title= 'Happiest Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Freedom',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
dat_2016 = df_2016.iloc[:3,:]

trace1 = go.Bar(
                x = dat_2016.country,
                y = df_2016.health,
                name = "Health",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dat_2016.country)

trace2 = go.Bar(
                x = dat_2016.country,
                y = dat_2016.economy,
                name = "Economy",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dat_2016.country)

trace3 = go.Bar(
                x = dat_2016.country,
                y = dat_2016.freedom,
                name = "Freedom",
                marker = dict(color = 'rgba(25, 205, 102, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dat_2016.country)

trace4 = go.Bar(
                x = dat_2016.country,
                y = dat_2016.family,
                name = "Family",
                marker = dict(color = 'rgba(125, 5, 72, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = dat_2016.country)
data = [trace1, trace2,trace3,trace4]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)
dat_2016 = df_2016.iloc[:3,:]

x = dat_2016.country

trace1 = {
  'x': x,
  'y': dat_2016.health,
  'name': 'Health',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': dat_2016.freedom,
  'name': 'Freedom',
  'type': 'bar'
};
trace3 = {
  'x': x,
  'y': dat_2016.economy,
  'name': 'Economy',
  'type': 'bar'
};
data = [trace1, trace2,trace3];
layout = {
  'xaxis': {'title': 'Top 3 Country'},
  'barmode': 'relative',
  'title': 'Health, freedom and economy ranking in the first 3 countries in 2016'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
dat_2016 = df_2016.iloc[:7,:]
pie1 = dat_2016.freedom
labels = dat_2016.country

fig = {
  "data": [
    {
      "values": pie1,
      "labels": labels,
      "domain": {"x": [0, .8]},
      "name": "Number Of Freedom Rates",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Country Number of freedom rates",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Number of Freedom",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
dat_2016 = df_2016.iloc[:20,:]
data = [
    {
        'y': dat_2016.freedom,
        'x': dat_2016.happiness_rank,
        'mode': 'markers',
        'marker': {
            'color': dat_2016.health,
            'size': dat_2016.happiness_score,
            'showscale': True
        },
        "text" :  dat_2016.country    
    }
]
iplot(data)
dfk2015 = df_2015.iloc[:100,:]
dfk2016 = df_2016.iloc[:100,:]
dff2015 = dfk2015.freedom
dff2016 = dfk2016.freedom
trace1 = go.Histogram(
    x=dff2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(171, 50, 36, 0.6)'),
    text=dfk2015.country)
    
trace2 = go.Histogram(
    x=dff2016,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(12, 50, 196, 0.6)'),
    text=dfk2016.country)

data = [trace1, trace2]

layout = go.Layout(barmode='overlay',
                  
                   title=' Freedom ratio in 2011 and 2012',
                   xaxis=dict(title='freedom-staff ratio'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
dfk2016 = df_2016.iloc[:100,:]
dff2016 = dfk2016.region

plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(dff2016))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
trace0 = go.Box(
    y=df_2016.economy,
    name = 'contribution margin of economy in 2016',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df_2016.health,
    name = 'contribution margin of health in 2016',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)

trace2 = go.Box(
    y=df_2016.freedom,
    name = 'contribution margin of freedom in 2016',
    marker = dict(
        color = 'rgb(102, 120, 140)',
    )
)

data = [trace0, trace1, trace2]
iplot(data)

dfk_2016 = df_2016.loc[:,["economy","health", "happiness_rank"]]
dfk_2016["index"] = np.arange(1,len(dfk_2016)+1)

fig = ff.create_scatterplotmatrix(dfk_2016, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat', text=df_2016.country,
                                  height=700, width=700)
iplot(fig)
trace1 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.freedom,
    name = "freedom",
    mode= "lines+markers",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
    text=df_2016.country
)

trace2 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.trust,
    xaxis='x2',
    yaxis='y2',
    name = "trust",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
    text=df_2016.country
)
data = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.6, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 0.95],
        anchor='x2',
    ),
    title = 'Trust and freedom in world countries'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Scatter3d(
    x=df_2016.happiness_rank,
    y=df_2016.economy,
    z=df_2016.health,
    text=df_2016.country,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(200,0,50)',                    
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=5,
        r=5,
        b=5,
        t=5  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
trace1 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.freedom,
    name = "freedom",
    text=df_2016.country
)
trace2 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.trust,
    xaxis='x2',
    yaxis='y2',
    name = "trust",
    text=df_2016.country
)
trace3 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.health,
    xaxis='x3',
    yaxis='y3',
    name = "health",
    text=df_2016.country
)
trace4 = go.Scatter(
    x=df_2016.happiness_rank,
    y=df_2016.economy,
    xaxis='x4',
    yaxis='y4',
    name = "economy",
    text=df_2016.country
)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'World Ranking of the countries with the highest score for health, trust, freedom and economy'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)