import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# matplotlib library

import matplotlib.pyplot as plt



# plotly library

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



# word cloud library

from wordcloud import WordCloud





import os

print(os.listdir("../input"))

data = pd.read_csv('../input/timesData.csv')
data.head()
data.info()
#The Analyzing of Research and Teaching of the Top 100 Universities



#Creating a data frame which includes a list of top 100 universities

df = data.iloc[:100]

#Creating a data1 which is list of the Research

data1 = go.Scatter(

    x = df.world_rank,

    y = df.research,

    mode = 'lines+markers',

    name = 'Research',

    marker = dict(color = 'rgba(45, 45, 240, 0.8)'),

    text = df.university_name )



#Creating a data2 which is list of the Research

data2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    mode = 'lines+markers',

    name = 'Teaching',

    marker = dict(color = 'rgba(255, 45, 45, 0.8)'),

    text = df.university_name)



merged_data = [data1, data2]

layout = dict(title = 'Research and Teaching of Top 100 Universities', xaxis = dict(title = 'World Rank', ticklen = 20, zeroline = False))



figure = dict(data = merged_data, layout = layout)

iplot(figure)
#The analyzing of the international of the top 100 universities according to years

#Creating Data Frames

df2014 = data[data.year == 2014].iloc[:100]

df2015 = data[data.year == 2015].iloc[:100]

df2016 = data[data.year == 2016].iloc[:100]



data2014 = go.Scatter(

    x = df2014.world_rank,

    y = df2014.international,

    mode = 'markers',

    name = '2014',

    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

    text = df2014.university_name

)

data2015 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.international,

    mode = 'markers',

    name = '2015',

    marker = dict(color = 'rgba(120, 128, 250, 0.8)'),

    text = df2015.university_name

)

data2016 = go.Scatter(

    x = df2016.world_rank,

    y = df2016.international,

    mode = 'markers',

    name = '2016',

    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

    text = df2016.university_name

)



merged_data = [data2014, data2015, data2016]

layout = dict(

    title = 'The Rates of the International in 3 years',

    xaxis = dict(title = 'World Rank', ticklen = 10, zeroline = False),

    yaxis = dict(title = 'Internatioal Rate', ticklen = 5, zeroline = False)

)



figure = dict(data = merged_data, layout = layout)

iplot(figure)
df2015 = data[data.year == 2015].iloc[:3]

df2015
#Teaching and Research of the top 5 universities

df2014 = data[data.year == 2014].iloc[:5]



data1 = go.Bar(

    x = df2014.university_name,

    y = df2014.teaching,

    name = 'Teaching',

    marker = dict(color = 'rgba(25, 25, 250, 0.6)'),

    text = df2014.country

)

data2 = go.Bar(

    x = df2014.university_name,

    y = df2014.research,

    name = 'Research',

    marker = dict(color = 'rgba(255, 15, 15, 0.6)'),

    text = df2014.country

)



merged_data = [data1, data2]

layout = go.Layout(barmode = 'group', title = 'Teaching and Research of the top 3 universities')

fig = go.Figure(data = merged_data, layout = layout)

iplot(fig)
#The same example with another kind of Bar Plot

df2014 = data[data.year == 2014].iloc[:5]

names = df2014.university_name



data1 = {

    "x": names,

    "y": df2014.teaching,

    "name": "Teaching",

    "type": "bar"

}

data2 = {

    'x': names,

    'y': df2014.research,

    'name': 'Research',

    'type': 'bar'

}

merged_data = [data1, data2]

layout = {

    'xaxis' : {'title': 'Top 5 Universities'},

    'barmode' : 'relative',

    'title' : 'Teaching and Research of the top 5 universities'

}

figure = go.Figure(data = merged_data, layout = layout)

iplot(figure)
df2016 = data[data.year == 2016].iloc[:5]

df2016
#Students rate of top 5 universities in 2016

df2016 = data[data.year == 2016].iloc[:5]

numbers = df2016.num_students

num_list = [float(each.replace(',','.')) for each in df2016.num_students]



figure = {

    'data': [{

        'values': num_list,

        'labels': df2016.university_name,

        'domain': {'x': [0, .9]},

        'name': 'Number of Student Rates',

        'hoverinfo': 'label+percent',

        'hole': .3,

        'type': 'pie'

    }],

    'layout': {

        'title': 'University Rates',

        'annotations': [{

            'font': {'size': 20},

            'showarrow': False,

            'text': 'Number Of Student',

            'x': 0.32,

            'y': 1.10

        }]  

    }

}

iplot(figure)
data.head(30)
#University world rank (first 30) vs Research score with number of students(size) and international score (color) in 2016

df2016 = data[data.year == 2016].iloc[:30, :]

df2016['num_students'].dropna(inplace = True)

num_list = [float(each.replace(',','.')) for each in df2016.num_students]

intern_color = [float(each) for each in df2016.international]



trace = [{

    'x': df2016.research,

    'y': df2016.world_rank,

    'mode': 'markers',

    'marker':{

        'color': intern_color,

        'size': num_list,

        'showscale': True

    },

    'text': df2016.university_name 

}]

iplot(trace)
data.head()
#Student Staff Ratio according to 2011 and 2012



data2011 = data.student_staff_ratio[data.year == 2011]

data2012 = data.student_staff_ratio[data.year == 2012]



dt1 = go.Histogram(

    x = data2011,

    opacity = 0.8,

    name = '2011',

    marker = dict(color = 'rgba(171, 50, 96, 0.7)')

)

dt2 = go.Histogram(

    x = data2012,

    opacity = 0.8,

    name = '2012',

    marker = dict(color = 'rgba(12, 50, 196, 0.7)')

)

merged_data = [dt1, dt2]

layout = go.Layout(

    barmode = 'overlay',

    title = 'Students Staff Ratio in 2011 and 2012',

    xaxis = dict(title = 'Student Staff Ratio'),

    yaxis = dict(title = 'Count')

)

figure = go.Figure(data = merged_data, layout = layout)

iplot(figure)
data.head()
#Total Score of Universities



data2015 = data[data.year == 2015]

trace1 = go.Box(

    y = data2015.total_score,

    name = 'Total Score of Universities',

    marker = dict(color = 'rgb(12, 12, 250)')

) 

trace2 = go.Box(

    y = data2015.income,

    name = 'Income of universities',

    marker = dict(color = 'rgb(250, 25, 25)')

)

merged_data = [trace1, trace2]

iplot(merged_data)
#The Relations between the Research, International and Teaching in 2015



#Preparing Data

df = data[data.year == 2015]

data2015 = df.loc[:, ['teaching', 'research', 'international']]

data2015['index'] = np.arange(1,len(data2015)+1)



#Using Scatter 

figure = ff.create_scatterplotmatrix(data2015, diag = 'box', index = 'index', colormap = 'Picnic', colormap_type = 'cat', height = 700, width = 700)

iplot(figure)

data.head()
df = data[data.year == 2014]

data1 = go.Scatter(

    x = df.world_rank,

    y = df.research,

    name = 'Research',

    marker = dict(color = 'rgb(25, 150, 90, 0.8)')

) 

data2 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Citations',

    marker = dict(color = 'rgb(200, 60, 25, 0.8)')

)

merged_data = [data1, data2]

layout = go.Layout(

    xaxis2=dict(domain=[0.65, 0.99999], anchor='y2'),

    yaxis2=dict(domain=[0.6, 0.95], anchor='x2'),

    title = 'Citations and Income vs World Rank of Universities'

)

figure = go.Figure(data = merged_data, layout = layout)

iplot(figure)
data.head()
df = data[data.year == 2015]

abc = np.array(df.world_rank.head(10))

abc = abc.astype(float)
abc
df = data[data.year == 2015]

abc = np.array(df.world_rank.head(201))

abc = abc.astype(float)

data2015 = go.Scatter3d(

    x = df.world_rank,

    y = df.citations,

    z = df.income,

    mode = 'markers',

    marker=dict(

        size=12,

        color=abc,                # set color to an array/list of desired values

        colorscale = 'Blues'  # choose a colorscale

    )

)



merged_data = [data2015]

layout = go.Layout(

    margin = dict(

        l = 0,

        r = 0,

        b = 0,

        t = 0

    )

)

figure = go.Figure(data = merged_data, layout = layout)

iplot(figure)



# x = World Rank of the Universities

# y = Citations

# x = Income
data.head()
# Total Score, Research, Citations and Teaching VS World Rank of Universities

df = data[data.year == 2015]



data1 = go.Scatter(

    x = df.world_rank,

    y = df.total_score,

    name = 'Total Score'

)

data2 = go.Scatter(

    x = df.world_rank,

    y = df.research,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Research'

)

data3 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    xaxis = 'x3',

    yaxis = 'y3',

    name = 'Citations'

)

data4 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    xaxis = 'x4',

    yaxis = 'y4',

    name = 'Teaching'

)



merged_data = [data1, data2, data3, data4]

layout = go.Layout(

    xaxis = dict(domain = [0, 0.45]),

    yaxis = dict(domain = [0, 0.45]),

    xaxis2 = dict(domain = [0.55, 1]),

    yaxis2 = dict(domain=[0, 0.45], anchor='x2'),

    xaxis3 = dict(domain=[0, 0.45], anchor='y3'),

    yaxis3 = dict(domain=[0.55, 1]),

    xaxis4 = dict(domain=[0.55, 1], anchor='y4'),

    yaxis4 = dict(domain=[0.55, 1], anchor='x4'),

    title = 'Total Score, Research, Citations and Teaching VS World Rank of Universities'

)



figure = go.Figure(data = merged_data, layout = layout)

iplot(figure)