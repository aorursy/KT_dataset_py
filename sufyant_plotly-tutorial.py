#pip install plotly
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Load data that we will use.

timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
timesData.shape
timesData.head()
timesData.info()
timesData.isna().sum()
timesData.student_staff_ratio.value_counts()
timesData.info()
# prepare data frame

df = timesData.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = "lines",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.university_name)

data = [trace1, trace2]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]
fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=df2014['world_rank'], y=df2014['citations'],

                    mode='markers',

                    name='2014',

                    text= df2014.university_name))

fig.add_trace(go.Scatter(x=df2015['world_rank'], y=df2015['citations'],

                    mode='markers',

                    name='2015',

                    text= df2015.university_name))

fig.add_trace(go.Scatter(x=df2016['world_rank'], y=df2016['citations'],

                    mode='markers',

                    name='2016',

                    text= df2016.university_name))

# Add title

fig.update_layout(

    title="Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years",

    xaxis_title="World Rank",

    yaxis_title="Citation")





fig.show()
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

df2014
fig = go.Figure(data=[

    go.Bar(name='Citations', x=df2014.university_name, y=df2014.citations,text = df2014.country),

    go.Bar(name='Teaching', x=df2014.university_name, y=df2014.teaching,text = df2014.country)

])

# Change the bar mode

fig.update_layout(barmode='group')

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Citations', x=df2014.university_name, y=df2014.citations),

    go.Bar(name='Teaching', x=df2014.university_name, y=df2014.teaching)

])

# Change the bar mode

fig.update_layout(barmode='stack')



# Change the title

fig.update_layout(

    title="Citations and teaching of top 3 universities in 2014",

    xaxis_title="Top 3 universities")



fig.show()
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

df2016
labels = df2016.university_name

values = [float(i.replace(',','.')) for i in df2016.num_students]
# Use `hole` to create a donut-like pie chart

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()
# pull is given as a fraction of the pie radius

fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])

fig.show()
df2016 = timesData[timesData.year == 2016].iloc[:20,:]

df2016
df2016.info()
# data preparation

df2016['world_rank'] = df2016['world_rank'].astype(float)

df2016['teaching'] = df2016['teaching'].astype(float)

df2016['international'] = df2016['international'].astype(float)

df2016['num_students'] = [float(i.replace(',','.')) for i in df2016.num_students]
fig = go.Figure(data=[go.Scatter(

    x=df2016['world_rank'],

    y=df2016['teaching'],

    text=df2016.university_name,

    mode='markers',

    marker=dict(

        color=df2016['international'],

        size=df2016['num_students'],

        showscale=True))])



fig.show()
# data preparation

df2011 = timesData[timesData.year == 2011]

df2012 = timesData[timesData.year == 2012]
fig = go.Figure()

fig.add_trace(go.Histogram(x=df2011.student_staff_ratio,name='2011'))

fig.add_trace(go.Histogram(x=df2012.student_staff_ratio,name='2012'))



# Overlay both histograms

fig.update_layout(barmode='overlay')



fig.update_layout(

    title_text='Students-staff ratio in 2011 and 2012', # title of plot

    xaxis_title_text='Student-staff ratio', # xaxis label

    yaxis_title_text='Count') # yaxis label

    

# Reduce opacity to see both histograms

fig.update_traces(opacity=0.75)

fig.show()
# data prepararion

x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# data preparation

x2015 = timesData[timesData.year == 2015]
fig = go.Figure()

fig.add_trace(go.Box(y=x2015.total_score, name='Total score of universities in 2015',

                marker_color = 'indianred'))

fig.add_trace(go.Box(y=x2015.research, name = 'Research of universities in 2015',

                marker_color = 'lightseagreen'))



fig.show()
# prepare data

dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international", "total_score"]]

data2015.index = np.arange(1, len(data2015)+1)

data2015["index"] = np.arange(1,len(data2015)+1)
data2015
# import figure factory

import plotly.figure_factory as ff



# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
import plotly as py

import plotly.graph_objs as go



trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name = "teaching")



trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',

    yaxis='y2',

    name = "income")



data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2'),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2'),    

    title = 'Income and Teaching vs World Rank of Universities')



fig = go.Figure(data=data, layout=layout)

iplot(fig)

dataframe
import plotly.express as px



fig = px.scatter_3d(dataframe, x=dataframe.world_rank,

                        y=dataframe.research,

                        z=dataframe.citations)

fig.show()
import plotly.graph_objects as go

from plotly.subplots import make_subplots



# Initialize figure with subplots

fig = make_subplots(rows=2, cols=2, start_cell="bottom-left")



# Initialize figure with subplots

fig = make_subplots(

    rows=2, cols=2, subplot_titles=("research", "citations", "income", "total_score")

)





# Add traces

fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.research,name='research'), row=1, col=1)

fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.citations,name='citations'), row=1, col=2)

fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.income,name='income'), row=2, col=1)

fig.add_trace(go.Scatter(x=dataframe.world_rank, y=dataframe.total_score,name='total_score'), row=2, col=2)



# Update yaxis properties

fig.update_yaxes(title_text="research", row=1, col=1)

fig.update_yaxes(title_text="citations", row=1, col=2)

fig.update_yaxes(title_text="income", row=2, col=1)

fig.update_yaxes(title_text="total_score", row=2, col=2)



# Update title and height

fig.update_layout(title_text="Research, citation, income and total score VS World Rank of Universities", height=700)





fig.show()