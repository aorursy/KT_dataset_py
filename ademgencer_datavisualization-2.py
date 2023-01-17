# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
timesdata = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")

timesdata.head()
timesdata.info()
# Prepare data for analysis

df100 = timesdata.head(100)



# Build traces

trace1 = go.Scatter(x = df100.world_rank,

                    y = df100.citations,

                    mode = "lines",

                    name = "citations rank",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df100.university_name)

trace2 = go.Scatter(x = df100.world_rank,

                    y = df100.teaching,

                    mode = "lines+markers",

                    name = "teaching rank",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df100.university_name)

tracedata = [trace1, trace2]



# Create a layout for graph

layout = dict(title = 'World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )



# Create figure dictionary

fig = dict(data=tracedata, layout = layout)



# Show graph

iplot(fig)
# Prepare data for analysis

#df2014 = timesdata[timesdata["year"] == 2014].head(100)     #Alternative for taking first 100 row

df2014 = timesdata[timesdata.year == 2014].iloc[:100,:]

df2015 = timesdata[timesdata.year == 2015].iloc[:100,:]

df2016 = timesdata[timesdata.year == 2016].iloc[:100,:]



# Create traces for graph

trace1 = go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(0, 128, 255, 0.8)'),

                    text= df2014.university_name)

trace2 = go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2015.university_name)

trace3 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(255, 128, 0, 0.8)'),

                    text= df2016.university_name)



# Combine traces

traces = [trace1, trace2, trace3]



# Configure layout

layout = dict(title = 'Citations of universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)

             )



# Create fig dictionary

fig = dict(data=traces, layout=layout)



# Draw chart

iplot(fig)
# Prepare data for analysis

df2014_f3 = df2014.iloc[:3,:]



# Create traces

trace1 = go.Bar(

                    x = df2014_f3.university_name,

                    y = df2014_f3.teaching,

                    name = "Teaching",

                    marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                    text= df2014.country)





trace2 = go.Bar(

                    x = df2014_f3.university_name,

                    y = df2014_f3.citations,

                    name = "Citations",

                    marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                    text= df2014.country)



# Combine traces

traces = [trace1, trace2]



# Configure layout

layout = go.Layout(barmode = "relative")     #barmode = group for grouping...



#Create fig

fig = go.Figure(data=traces, layout=layout)



# Create graph

iplot(fig)
df2016_f7.head()
# Prepare data

df2016_f7 = df2016.iloc[:7,:]



# Prepare value list

value_list = [float(each.replace(',', '.')) for each in df2016_f7.num_students]



# Create trace

trace1 = {

      "values": value_list,

      "labels": df2016_f7.university_name,

      "domain": {"x": [0, .9]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .4,

      "type": "pie"

    }



# Create layout

layout = dict(title="University students", 

              annotations= [{ "font": { "size": 20}, "showarrow": True, "text": "Number of Students", "x": 0.20, "y": 1 }])



# Create pie figure

fig = dict(data=trace1, layout=layout)



# Show graph

iplot(fig)
# Prepare data 

df2016_f20 = timesdata[timesdata.year == 2016].iloc[:20,:]



# Define traces

numof_students  = [float(each.replace(',', '.')) for each in df2016_f20.num_students]

numof_international  = [float(each.replace(',', '.')) for each in df2016_f20.international]



# define data

data = [

    {

        'y': df2016_f20.teaching,

        'x': df2016_f20.world_rank,

        'mode': 'markers',

        'marker': {

            'color': numof_international,

            'size': numof_students,

            'showscale': True

        },

        "text" :  df2016.university_name    

    }

]



# Draw graph

iplot(data)
# Prepare data for analysis

x2011 = timesdata.country[timesdata.year == 2011]

x2012 = timesdata.country[timesdata.year == 2012]



# Create traces

trace1 = go.Histogram(x=x2011, opacity=.5, name="2011")

trace2 = go.Histogram(x=x2012, opacity=.5, name="2012")



# Combine traces

traces = [trace1, trace2]

# Create layout

layout = go.Layout(barmode="overlay", title="Countries", xaxis=dict(title="Countries"), yaxis=dict(title="Count"))



# Create fig

fig = go.Figure(data=traces, layout=layout)



# Draw graph

iplot(fig)
# Data preperation

x2015 = timesdata[timesdata.year == 2015]

x2015.head()



# Create traces

trace1 = go.Box(y=x2015.research, name="Research")

trace2 = go.Box(y=x2015.total_score, name="Total score")



# Draw graph

iplot([trace1, trace2])
# import figure factory

import plotly.figure_factory as ff
# Prepare data

df = timesdata[timesdata.year == 2016].loc[:,["research", "citations", "total_score"]]

df["index"] = np.arange(1, len(df)+1)
# Create figure

fig = ff.create_scatterplotmatrix(df, colormap="Portland", colormap_type="cat", index="index")



# Draw graph

iplot(fig)
plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(timesdata.country))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()