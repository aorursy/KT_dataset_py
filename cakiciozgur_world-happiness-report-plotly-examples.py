# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

#wordcloud
from wordcloud import WordCloud

#matplotlib
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data2015=pd.read_csv("../input/2015.csv")
data2016=pd.read_csv("../input/2016.csv")
data2017=pd.read_csv("../input/2017.csv")
#data2015.info()
data2015.columns = data2015.columns.str.strip().str.replace(' ', '_').str.lower().str.replace("(","").str.replace(")","")
data2016.columns = data2016.columns.str.strip().str.replace(' ', '_').str.lower().str.replace("(","").str.replace(")","")
data2017.columns = data2017.columns.str.strip().str.replace('.', '_').str.lower()
#data2015.head()
#data2016.head()
#data2017.head()
datax100=data2015.iloc[:100,:]
datax101=data2016.iloc[:100,:]
datax102=data2017.iloc[:100,:]
#1
trace0=go.Scatter(
    x=datax100.happiness_rank,
    y=datax100.health_life_expectancy,
    mode="lines+markers",
    name="HeaaltLife",
    marker=dict(color="rgba(255, 112, 156, 0.99)"),
    text=datax100.region
)
trace1=go.Scatter(
    x=datax100.happiness_rank,
    y=datax100.trust_government_corruption,
    mode="lines+markers",
    name="TrustGovernment",
    marker=dict(color="rgba(20, 11, 15, 0.5)"),
    text=datax100.region
)
dataline=[trace0,trace1]
layout=dict(
    title="Heaalt Life Expectancy and Trust Government Corruption of Top 100 Country",
    xaxis=dict(title="Country",ticklen=2,zeroline=True)
)
fig=go.Figure(data=dataline,layout=layout)
iplot(fig)

#1
trace0=go.Scatter(
    x=datax100.happiness_rank,
    y=datax100.freedom,
    mode="lines+markers",
    name="2015",
    marker=dict(color="rgba(255, 230, 117, 0.99)"),
    text=datax100.country
)
trace1=go.Scatter(
    x=datax101.happiness_rank,
    y=datax101.freedom,
    mode="lines+markers",
    name="2016",
    marker=dict(color="rgba(175, 112, 112, 0.99)"),
    text=datax101.country
)
trace2=go.Scatter(
    x=datax102.happiness_rank,
    y=datax102.freedom,
    mode="lines+markers",
    name="2017",
    marker=dict(color="rgba(75, 120, 0, 0.99)"),
    text=datax102.country
)
datascatter=[trace0,trace1,trace2]
layout = dict(title = 'Freedom vs Happiness Rank of Top 100 Country with 2015, 2016 and 2017 years',
              xaxis= dict(title= 'Happiness Rank',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Freedom',ticklen= 5,zeroline= False)
             )
fig=go.Figure(data=datascatter,layout=layout)
iplot(fig)
#datax3
datax3=data2015.iloc[:3]
#1
#Normalization
datax3["hs"]=(datax3.happiness_score)/max(datax3.happiness_score)
datax3["fd"]=(datax3.freedom)/max(datax3.freedom)

trace0=go.Bar(
    x=datax3.country,
    y=datax3.hs,
    name = "Happines Score",
                marker = dict(color = 'rgba(255, 40, 140, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = datax3.region+" "+datax3.country)
trace1=go.Bar(
    x=datax3.country,
    y=datax3.fd,
    name = "Freedom",
                marker = dict(color = 'rgba(120, 140, 125, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = datax3.region+" "+datax3.country)

databar=[trace0,trace1]
layout=dict(title="The Happiness Rates and Freedom Rates of The First 3 Cities in 2015",barmode="stack")
fig=go.Figure(data=databar,layout=layout)
iplot(fig)
#2
#Normalization
datax3["egpc"]=(datax3.economy_gdp_per_capita)/max(datax3.economy_gdp_per_capita)

trace0=go.Bar(
    x=datax3.country,
    y=datax3.hs,
    name = "Happines Score",
                marker = dict(color = 'rgba(255, 40, 140, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = datax3.region+" "+datax3.country)
trace1=go.Bar(
    x=datax3.country,
    y=datax3.egpc,
    name = "Economy gdp per capita",
                marker = dict(color = 'rgba(120, 140, 125, 0.8)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = datax3.region+" "+datax3.country)

databar=[trace0,trace1]
layout=dict(title="The Happiness Rates and Economy gdp per capita of The First 3 Cities in 2015",barmode="group")
fig=go.Figure(data=databar,layout=layout)
iplot(fig)
#1
datax7=data2016.iloc[:7]
labels=datax7.country
values=datax7["happiness_score"].iloc[:7]

fig = {
  "data": [
    {
      "values": values,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Happiness Score",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"The happiness ranks of the first 7 cities in 2016",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": True,
              "text": "Happines Rank",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}
iplot(fig)
#1
datax15=data2016.iloc[:15]
size_x15=datax15.happiness_score*2
color_x15=datax15.freedom

databubble = [
    {
        "x":datax15.happiness_rank,
        "y":datax15.upper_confidence_interval,
        "mode":"markers",
        "marker":{
            "color":color_x15,
            "size":size_x15,
            "showscale":True
        },
        "text":datax15.country
    }
]
iplot(databubble)

#1
datax2016=data2016.copy()
datax2017=data2017.copy()

trace1=go.Histogram(
            x=datax2016.happiness_score,
            opacity=0.9,
            name="2016",
            marker=dict(color="rgba(255,30,54,0.9)")
)
trace2=go.Histogram(
            x=datax2017.happiness_score,
            opacity=0.5,
            name="2017",
            marker=dict(color="rgba(1,120,255,0.5)")
)
datahist=[trace1,trace2]
layout=go.Layout(title="Happiness Score in 2011 and 2012",
                xaxis=dict(title="Happiness Score"),
                yaxis=dict(title="Frequency"),
                barmode="group")
fig=go.Figure(data=datahist,layout=layout)
iplot(fig)
#1
plt.subplots(figsize=(10,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(data2015.region))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
#1
datax100=data2017.iloc[:100]

trace0=go.Box(
    y=datax100.whisker_high,
    name="Country of whisker high",
    marker=dict(color="rgba(233,0,0,0.7)")
)
trace1=go.Box(
    y=datax100.whisker_low,
    name="Country of whisker low",
    marker=dict(color="rgba(0,0,150,0.7)")
)
databox=[trace0,trace1]
iplot(databox)
#1
import plotly.figure_factory as ff  #freedom, generosity, happiness_score
dataxFeature=data2015.loc[:,["freedom","generosity","happiness_score"]]
dataxFeature["index"]=np.arange(1,len(dataxFeature)+1)

fig=ff.create_scatterplotmatrix(
                                  dataxFeature, diag='box',
                                  index='index',
                                  colormap='Viridis',
                                  colormap_type='cat',
                                  height=700, width=700
                               )
iplot(fig)
#1
trace1 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.happiness_score,
    name = "Happiness Score",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
trace2 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.economy_gdp_per_capita,
    xaxis='x2',
    yaxis='y2',
    name = "Economy gdp per capita",
    marker = dict(color = 'rgba(255, 112, 20, 0.8)'),
)
datainset = [trace1, trace2]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.4, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.6, 1],
        anchor='x2',
    ),
    title = 'Happiness score and economy gdp per capita of Top 100 Country'
)
fig = go.Figure(data=datainset, layout=layout)
iplot(fig)
#1
trace1 = go.Scatter3d(
    x=data2015.happiness_rank,
    y=data2015.happiness_score,
    z=data2015.freedom,
    mode='markers',
    marker=dict(
        size=7.5,
        color='rgb(52,150,255)',        
    )
)
data3dscatter=[trace1]
layout=go.Layout(
        margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
)
fig=go.Figure(data=data3dscatter,layout=layout)
iplot(fig)
#1   #happines_score, freedom, economy_gdp_per_capita, generosity
trace1 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.happiness_score,
    name = "Happines Score"
)
trace2 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.freedom,
    xaxis='x2',
    yaxis='y2',
    name = "Freedom"
)
trace3 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.economy_gdp_per_capita,
    xaxis='x3',
    yaxis='y3',
    name = "Economy gdp per capita"
)
trace4 = go.Scatter(
    x=data2016.happiness_rank,
    y=data2016.generosity,
    xaxis='x4',
    yaxis='y4',
    name = "Generosity"
)
datamultiple=[trace1,trace2,trace3,trace4]
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
    title = "Happines Score, Freedom, Economy gdp per capita and Generosity of Country"
)
fig=go.Figure(data=datamultiple,layout=layout)
iplot(fig)