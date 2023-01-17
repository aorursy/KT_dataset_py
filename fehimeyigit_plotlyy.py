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

# word cloud library
from wordcloud import WordCloud

# matplotlib
import matplotlib.pyplot as plt


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load data that we will use.
data_2017 = pd.read_csv('../input/2017.csv')
data_2016 = pd.read_csv('../input/2016.csv')
data_2015= pd.read_csv('../input/2015.csv')
# information about data_2017
data_2017.info
data_2017.columns
data_2017.rename(
    columns={
   'Happiness.Score' : 'Happiness_Score',
   'Happiness.Rank' : 'Happiness_Rank',
   'Whisker.high' : 'Whisker_high',
   'Whisker.low' : 'Whisker_low',
   'Economy..GDP.per.Capita.' : 'Economy',
   'Health..Life.Expectancy.' : 'Health_Life_Expectancy',
   'Trust..Government.Corruption.' : 'Trust_Government_Corruption',
   'Dystopia.Residual' : 'Dystopia_Residual'
   
  },
  inplace=True
)
data_2017.columns
data_2016.columns
data_2016.rename(
    columns={
   'Happiness Score' : 'Happiness_Score',
   'Happiness Rank' : 'Happiness_Rank',
   'Lower Confidence Interval' : 'Lower_Confidence_Interval',
   'Upper Confidence Interval' : 'Upper_Confidence_Interval',
   'Economy (GDP per Capita)' : 'Economy',
   'Health (Life.Expectancy)' : 'Health_Life_Expectancy',
   'Trust (Government.Corruption)' : 'Trust_Government_Corruption',
   'Dystopia Residual' : 'Dystopia_Residual'
   
  },
  inplace=True
)
data_2016.columns
data_2015.columns
data_2015.rename(
    columns={
   'Happiness Score' : 'Happiness_Score',
   'Happiness Rank' : 'Happiness_Rank',
   'Standard Error' : 'Standard_Error',
   'Economy (GDP per Capita)' : 'Economy',
   'Health (Life Expectancy)' : 'Health_Life_Expectancy',
   'Trust (Government.Corruption)' : 'Trust_Government_Corruption',
   'Dystopia Residual' : 'Dystopia_Residual'
   
  },
  inplace=True
)
data_2015.columns
data_2017.head(10)
data_2017["Happiness_Score"].value_counts(dropna =False)
data_2016["Happiness_Score"].value_counts(dropna =False)
data_2015["Happiness_Score"].value_counts(dropna =False)
 # prepare data frame
df = data_2017.iloc[:150,:]

# import graph objects as "go"
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    #x = x axis
                    x = df.Happiness_Score,
                    #y = y axis
                    y = df.Economy,
                    #  mode = type of plot like marker, line or line + markers
                     mode = "lines",
                    #name = name of the plots
                     name = "Economy",
                     #marker = marker is used with dictionary.
                     #color = color of lines. It takes RGB (red, green, blue) and opacity (alpha)
                     marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
                     #text = The hover text (hover is curser)
                    text= df.Country)
# Creating trace2
trace2 = go.Scatter(
                    x = df.Happiness_Score,
                    y = df.Family,
                    mode = "lines+markers",
                    name = "Family",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
                    text= df.Country)


#data = is a list that we add traces into it
data = [trace1, trace2]
#layout = it is dictionary.
#title = title of layout
#x axis = it is dictionary
#title = label of x axis
#ticklen = length of x axis ticks
#zeroline = showing zero line or not
layout = dict(title = 'Economy and Family vs Happiness Score of  150 Countries',
              xaxis= dict(title= 'Happiness Score',ticklen= 5,zeroline= False)
             )

#fig = it includes data and layout
fig = dict(data = data, layout = layout)
#iplot() = plots the figure(fig) that is created by data and layout
iplot(fig)
# prepare data frames
df2017 = data_2017.iloc[:150,:]
df2015 = data_2015.iloc[:150,:]
df2016 = data_2016.iloc[:150,:]

# import graph objects as "go"
import plotly.graph_objs as go
# creating trace1
trace1 =go.Scatter(
                    x = df2017.Happiness_Score,
                    y = df2017.Family,
                    mode = "markers",
                    name = "2017",
                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),
                    text= df2017.Country)
# creating trace2
trace2 =go.Scatter(
                    x = df2015.Happiness_Score,
                    y = df2015.Family,
                    mode = "markers",
                    name = "2015",
                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),
                    text= df2015.Country)
# creating trace3
trace3 =go.Scatter(
                    x = df2016.Happiness_Score,
                    y = df2016.Family,
                    mode = "markers",
                    name = "2016",
                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),
                    text= df2016.Country)
data = [trace1, trace2, trace3]
layout = dict(title = 'Happiness Score vs Family  of 150 Countries with 2015, 2016 and 2017 years',
              xaxis= dict(title= 'Happiness Score',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'Family',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)
# prepare data frames
df2017 = data_2017.iloc[:15,:]

# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df2017.Country,
                y = df2017.Happiness_Score,
                name = "Happiness_Score",
                marker = dict(color = 'rgba(255, 174, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2017.Country)
# create trace2 
trace2 = go.Bar(
                x = df2017.Country,
                y = df2017.Economy,
                name = "Economy",
                marker = dict(color = 'rgba(255, 255, 128, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = df2017.Country)
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)

# prepare data frames
df2016 = data_2016.iloc[:15,:]
# import graph objects as "go"
import plotly.graph_objs as go

x = df2016.Country

trace1 = {
  'x': x,
  'y': df2016.Happiness_Score,
  'name': 'Happiness Score',
  'type': 'bar'
};
trace2 = {
  'x': x,
  'y': df2016.Economy,
  'name': 'Economy',
  'type': 'bar'
};
data = [trace1, trace2];
layout = {
  'xaxis': {'title': 'Countries'},
  'barmode': 'relative',
  'title': 'Happiness Score and Economy of top 15 Countries in 2016'
};
fig = go.Figure(data = data, layout = layout)
iplot(fig)
# import graph objects as "go" and import tools
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt
# prepare data frames
df2015 = data_2015.iloc[:15,:]

y_saving = [each for each in df2015.Happiness_Score]
y_net_worth  = [float(each) for each in df2015.Economy]
x_saving = [each for each in df2015.Country]
x_net_worth  = [each for each in df2015.Country]
trace0 = go.Bar(
                x=y_saving,
                y=x_saving,
                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),
                name='Happiness Score',
                orientation='h',
)
trace1 = go.Scatter(
                x=y_net_worth,
                y=x_net_worth,
                mode='lines+markers',
                line=dict(color='rgb(63, 72, 204)'),
                name='Economy',
)
layout = dict(
                title='Happiness Score and Economy',
                yaxis=dict(showticklabels=True,domain=[0, 0.85]),
                yaxis2=dict(showline=True,showticklabels=False,linecolor='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0, 0.85]),
                xaxis=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0, 0.42]),
                xaxis2=dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47, 1],side='top',dtick=25),
                legend=dict(x=0.029,y=1.038,font=dict(size=10) ),
                margin=dict(l=200, r=20,t=70,b=70),
                paper_bgcolor='rgb(248, 248, 255)',
                plot_bgcolor='rgb(248, 248, 255)',
)
annotations = []
y_s = np.round(y_saving, decimals=2)
y_nw = np.rint(y_net_worth)
# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x_saving):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2', y=xd, x=ydn - 4,text='{:,}'.format(ydn),font=dict(family='Arial', size=12,color='rgb(63, 72, 204)'),showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1', y=xd, x=yd + 3,text=str(yd),font=dict(family='Arial', size=12,color='rgb(171, 50, 96)'),showarrow=False))

layout['annotations'] = annotations

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                          shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(layout)
iplot(fig)
# data preparation
df2015 = data_2015
df2015.Region.value_counts()
# data preparation
df2015 = data_2015
values = df2015.Region.value_counts()
labels = ['Sub-Saharan Africa',  'Central and Eastern Europe', 'Latin America and Caribbean',
'Western Europe', 'Middle East and Northern Africa', 'Southeastern Asia', 'Southern Asia', 'Eastern Asia','Australia and New Zealand',
'North America']



# figure
fig = {
  "data": [
    {
      "values": values,
      "labels": labels,
      "domain": {"x": [0, .5]},
      "name": "Regions",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Regions of Countries ",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Percent of Regions",
                "x": 0.20,
                "y": 1
            },
        ]
    }
}

#trace = go.Pie(labels=labels, values=values, title="Regions of Countries")

iplot(fig)
# data preparation
# Firstly lets create 2 data frame
data1 = data_2015.iloc[:10,:]
data2= data_2015.iloc[80:90,:]
df2015 = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
#we used 2 parts of data to see differences between Countries happiness scores
#(scores are so close to each other and very difficult to understand differences  between 7.5620 - 7.4856)
size  = df2015.Happiness_Score
color = df2015.Economy
data = [
    {
        'y': df2015.Economy,
        'x': df2015.Country,
        'mode': 'markers',
        'marker': {
            'color': color,
            'size': size,
        
            'showscale': True
        },
        "text" :  df2015.Country    
    }
]
iplot(data)

     

# prepare data frames
df2017 = data_2017.Happiness_Score
df2015 = data_2015.Happiness_Score
df2016 = data_2016.Happiness_Score
trace1 = go.Histogram(
    x=df2017,
    opacity=0.75,
    name = "2017",
    marker=dict(color='rgba(171, 50, 96, 0.6)'))
trace2 = go.Histogram(
    x=df2016,
    opacity=0.75,
    name = "2016",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

trace3 = go.Histogram(
    x=df2015,
    opacity=0.75,
    name = "2015",
    marker=dict(color='rgba(12, 50, 196, 0.6)'))

data = [trace1, trace2,trace3]
layout = go.Layout(barmode='overlay',
                   title=' Happiness Scores in 2015,2016 and 2017',
                   xaxis=dict(title='Happiness Scores'),
                   yaxis=dict( title='Count'),
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
# data prepararion
df2017 = data_2017.Country
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=384
                         ).generate(" ".join(df2017))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()
# data preparation
df2015 = data_2015

trace0 = go.Box(
    y=df2015.Happiness_Score,
    name = 'Happiness score of countries in 2015',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
trace1 = go.Box(
    y=df2015.Economy,
    name = 'Economy of countries in 2015',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
data = [trace0, trace1]
iplot(data)
# import figure factory
import plotly.figure_factory as ff
# prepare data
dataframe = data_2015
data2015 = dataframe.loc[:,["Happiness_Score","Family", "Economy"]]
data2015["index"] = np.arange(1,len(data2015)+1)
# scatter matrix
fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',
                                  height=700, width=700)
iplot(fig)
# first line plot
trace1 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Happiness_Score,
    name = "Happiness_Score",
    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),
)
# second line plot
trace2 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Economy,
    xaxis='x2',
    yaxis='y2',
    name = "Economy",
    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),
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
    title = 'Economy and Happiness Score vs Happiness Rank of Countries'

)

fig = go.Figure(data=data, layout=layout)
iplot(fig)
# create trace 1 that is 3d scatter
trace1 = go.Scatter3d(
    x=dataframe.Happiness_Rank,
    y=dataframe.Happiness_Score,
    z=dataframe.Economy,
    mode='markers',
    marker=dict(
        size=10,
        color='rgb(255,0,0)',                # set color to an array/list of desired values      
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    )
    
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)

trace1 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Family,
    name = "Family"
)
trace2 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Health_Life_Expectancy,
    xaxis='x2',
    yaxis='y2',
    name = "Health"
)
trace3 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Economy,
    xaxis='x3',
    yaxis='y3',
    name = "Economy"
)
trace4 = go.Scatter(
    x=dataframe.Happiness_Rank,
    y=dataframe.Happiness_Score,
    xaxis='x4',
    yaxis='y4',
    name = "Happiness_Score"
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
    title = 'Family, Health, Economy and Happiness score VS Happiness Rank of Countries'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)