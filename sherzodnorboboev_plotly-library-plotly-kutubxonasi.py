# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv) | datani ustigaishlash uchun, csv fayllarni bilan ishlash

import matplotlib.pyplot as plt



# plotly

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected = True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt



# warnings

import warnings

warnings.filterwarnings('ignore')



# this is not so important, juts seeing what we have | bu juda muhim emas, shunchaki nima bor ekanligini ko'rish uchun

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load data that we will use | Foydalanadigan datani yuklash

timesData = pd.read_csv("../input/world-university-rankings/timesData.csv")
# Information about data | Data haqida ma'lumot

timesData.info()
# first 10 rows of data | dataning birinchi 10 qatori

timesData.head(10)
# prepare DataFrame | DataFrameni tayyorlash

df = timesData.iloc[:100,:]



# Creating trace | Trace(shunchaki element) yaratish

trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = 'lines',

                    name = 'citations',

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text = df.university_name

)

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = 'lines+markers',

                    name = 'teaching',

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text = df.university_name

)

data = [trace1, trace2]

layout = dict(title = "Citations and Teaching  vs World Rank of Top 100 Universities <br>Citations va Teaching ni top 100 ta universitet o'rni bo'yicha",

              xaxis = dict(title = 'World Rank', ticklen = 5, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare DataFrame | DataFrame ni tayyorlash

df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]



# Creating trace | Trace yaratish

trace1 = go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = 'markers',

                    name = '2014',

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text = df2014.university_name

)

trace2 = go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = 'markers',

                    name = '2015',

                    marker = dict(color = 'rgba(12, 128, 14, 0.8)'),

                    text = df2015.university_name

)

trace3 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = 'markers',

                    name = '2016',

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text = df2016.university_name

)

data = [trace1, trace2, trace3]

layout = dict(title = """Citations vs World Rank of Top 100 Universities with 2014, 2015 and 2016 years<br>

2014, 2015, 2016-yillarda 100 ta top universitetning Citations va World Ranki""",

              xaxis = dict(title = 'World Rank', ticklen = 10, zeroline = False),

              yaxis = dict(title = 'Citation', ticklen = 5, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# making figure )) | shakl yasash

# bow figure | kamon shakli

fig = go.Figure(data=[

    go.Scatter(

        x=[1, 10, 20, 30, 40, 50, 60, 70, 80],

        y=[80, 70, 60, 50, 40, 30, 20, 10, 1]

    ),

    go.Scatter(

        x=[1, 80],

        y=[80, 1]

    ),

    go.Scatter(

        x=[1, 10, 20, 30, 40, 50, 60, 70, 80],

        y=[1, 10, 20, 30, 40, 50, 60, 70, 80]

    )

])



fig.update_xaxes(type="log")

fig.update_yaxes(type="log")



fig.show()
# style 1

# prepare DataFrame | DataFrameni tayyorlash

df2014 = timesData[timesData.year == 2014].iloc[:3,:]



# Creating traces

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = 'citations',

                marker = dict(color = 'rgba(200, 174, 200, 0.5)',

                             line = dict(color = 'rgb(0, 0, 0)', width = 1.5)),

                text = df2014.country

)

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = 'teaching',

                marker = dict(color = 'rgba(12, 255, 128, 0.8)',

                             line = dict(color = 'rgb(0, 0, 0)', width = 1.5)),

                text = df2014.country

)

data = [trace1, trace2]

layout = dict(title = """Citations vs Teaching of Top 3 Universities in 2014 year<br>

2014-yilda Top 3 ta universitetlarning Citations va Teachinglari""",

              xaxis = dict(title = 'Top 3 Universities', ticklen = 10, zeroline = False),

              barmode = 'group'

             )

# layout = go.Layout(...) #you can use this, instead of dict | dict o'rniga ishlata olish mumkin

fig = dict(data = data, layout = layout)

iplot(fig)
# style 2

# prepare DataFrame | DataFrame tayyorlash

df2014 = timesData[timesData.year == 2014].iloc[:3,:]



# Creating traces

x = df2014.university_name



trace1 = {

    'x': x,

    'y': df2014.citations,

    'name': 'citations',

    'type': 'bar'

}

trace2 = {

    'x': x,

    'y': df2014.teaching,

    'name': 'citations',

    'type': 'bar'

}

data = [trace1, trace2]

layout = {

    'title': """Citations vs Teaching of Top 3 Universities in 2014 year<br>

2014-yilda Top 3 ta universitetlarning Citations va Teachinglari""",

    'xaxis': {'title': 'Top 3 Universities' },

    'barmode': 'relative'

}



fig = dict(data = data, layout = layout)

iplot(fig)
# style 3

from plotly import tools

from plotly import subplots 



# prepare DataFrame | DataFrame tayyorlash

df2016 = timesData[timesData.year == 2016].iloc[:7,:]



# Creating traces | Tracelar yaratish

y_saving = [each for each in df2016.research]

y_net_worth = [float(each) for each in df2016.income]

university_name2016 = [each for each in df2016.university_name]



trace0 = go.Bar(

                x = y_saving,

                y = university_name2016,

                marker = dict(color = 'rgba(200, 174, 200, 0.5)',

                             line = dict(color = 'rgb(0, 0, 0)', width = 1.5)),

                name = 'research',

                orientation = 'h'

)

trace1 = go.Scatter(

                x = y_net_worth,

                y = university_name2016,

                mode = 'lines+markers',

                marker = dict(color = 'rgba(12, 255, 128, 0.8)'),

                name = 'income'

)

data = [trace1, trace2]

layout = dict(

              title = 'Citations vs Income',

              yaxis = dict(showticklabels = True, domain = [0, 0.85]),

              yaxis2 = dict(showline = True, showticklabels = False, linecolor = 'rgba(50, 102, 102, 0.8)', linewidth = 2, domain = [0, 0.85]),

              xaxis = dict(zeroline = False, showline = False, showticklabels = True, showgrid = True, domain = [0, 0.45]),

              xaxis2 = dict(zeroline = False, showline = False, showticklabels = True, showgrid = True, domain = [0.47, 1], side = 'top', dtick = 5),

              legend = dict(x = 0.029, y = 1.038, font = dict(size = 10)),

              margin = dict(l = 200, r = 20, t = 70, b = 70),

              paper_bgcolor = 'rgb(248, 248, 255)',

              plot_bgcolor = 'rgb(248, 248, 255)'

)

annotations = []

y_s = np.round(y_saving, decimals = 2)

y_nw = np.rint(y_net_worth)



# Adding labels

check_number = 0

for ydn, yd, xd in zip(y_nw, y_s, university_name2016):

    # labelling the scatter savings | scatter savingsga label qo'yish

    annotations.append(dict(xref = 'x2', yref = 'y2', y = xd, x = (ydn - 2) if check_number > ydn else (ydn + 2), text = '{:,}'.format(ydn), font = dict(family = 'Arial', size = 12, color = 'rgb(255, 78, 45)'), showarrow=False))

    # labelling the bar net worth | net worth barga label qo'yish

    annotations.append(dict(xref = 'x1', yref = 'y1', y = xd, x = yd + 3, text = str(yd), font = dict(family = 'Arial', size = 12, color = 'rgb(171, 50, 96)'), showarrow=False))

    check_number = ydn



layout['annotations'] = annotations



#  Creating two subplots | 2 ta sublot yaratish

fig = tools.make_subplots(rows = 1, cols = 2, specs = [[{},{}]], shared_xaxes = True,

                        shared_yaxes = False, vertical_spacing = 0.001)



# fig = subplots.make_subplots(rows = 1, cols = 2, specs = [[{},{}]], shared_xaxes = True,

#                         shared_yaxes = False, vertical_spacing = 0.001)



fig.append_trace(trace0, 1, 1)

fig.append_trace(trace1, 1, 2)



fig['layout'].update(layout)

iplot(fig)
colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'cryn']



fig = go.Figure(data=[go.Pie(labels=['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen', 'Other'],

                             values=[4500,2500,1053,500,300],

                             title='Elements of air',

                             hole=.3,

                             pull=[0, 0, 0.2, 0, 0])])

fig.update_traces(hoverinfo='label+percent+value', textposition='inside', textinfo='label+percent', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=0.5)))

iplot(fig)

#  fig.show()
from plotly.subplots import make_subplots



labels = ['1st', '2nd', '3rd', '4th', '5th']



# Define color sets of paintings

night_colors = ['rgb(56, 75, 126)', # 1st

                'rgb(18, 36, 37)',  # 2nd

                'rgb(34, 53, 101)', # 3rd

                'rgb(36, 55, 57)',  # 4th

                'rgb(6, 4, 4)']     # 5th



sunflowers_colors = ['rgb(177, 127, 38)',  # 1st

                     'rgb(205, 152, 36)',  # 2nd

                     'rgb(99, 79, 37)',    # 3rd

                     'rgb(129, 180, 179)', # 4th

                     'rgb(124, 103, 37)']  # 5th



irises_colors = ['rgb(33, 75, 99)',    # 1st

                 'rgb(79, 129, 102)',  # 2nd

                 'rgb(151, 179, 100)', # 3rd

                 'rgb(175, 49, 35)',   # 4th

                 'rgb(36, 73, 147)']   # 5th



cafe_colors =  ['rgb(146, 123, 21)',  # 1st

                'rgb(177, 180, 34)',  # 2nd

                'rgb(206, 206, 40)',  # 3rd

                'rgb(175, 51, 21)',   # 4th

                'rgb(35, 36, 21)']    # 5th



# Create subplots, using 'domain' type for pie charts

specs = [[{'type':'domain'}, {'type':'domain'}], [{'type':'domain'}, {'type':'domain'}]]

fig = make_subplots(rows=2, cols=2, specs=specs)



# Define pie charts

fig.add_trace(go.Pie(labels=labels,

                     values=[38, 27, 18, 10, 7],

                     name='Starry Night',

                     marker_colors=night_colors),

              1,

              1)

fig.add_trace(go.Pie(labels=labels,

                     values=[28, 26, 21, 15, 10],

                     name='Sunflowers',

                     marker_colors=sunflowers_colors),

              1,

              2)

fig.add_trace(go.Pie(labels=labels,

                     values=[38, 19, 16, 14, 13],

                     name='Irises',

                     marker_colors=irises_colors),

              2,

              1)

fig.add_trace(go.Pie(labels=labels, 

                     values=[31, 24, 19, 18, 8], 

                     name='The Night Café',

                     marker_colors=cafe_colors), 

              2,

              2)



# Tune layout and hover info

fig.update_traces(hoverinfo='label+percent+name',

                  textinfo='none')

fig.update(layout_title_text='Van Gogh: 5 Most Prominent Colors Shown Proportionally',

           layout_showlegend=False)



fig = go.Figure(fig)

fig.show()
timesData.head(2)
# data preparation | data tayyorlash

df2016 = timesData[timesData.year == 2016].iloc[:7,:]

values = [float(each.replace(',', '.')) for each in df2016.num_students]

labels = df2016.university_name



# vizualization

fig = {

    'data': [{

        'labels': labels,

        'name': 'Number of Students rates',

      "domain": {"x": [0.5, .9]},

        'values': values,

        'hoverinfo': 'label+percent+value',

        'textposition': 'inside',

        'textinfo': 'percent',

        'pull': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],

        'hole': .25,

        'type': 'pie'

    }],

    'layout': {

        'title': 'Universities Number of Students rates',

        'annotations': [{

            "font": { "size": 20},

            "showarrow": False,

            "text": "Number of Students",

            "x": 0.15,

            "y": 1

        },]

    }

}

iplot(fig)
timesData.info()
timesData.head(2)
data = timesData[timesData.year == 2014].iloc[:10,:]

data.replace(['-'], 20, inplace = True)

data.income = data.income.astype('float')



# vizualization

import seaborn as sns

colors = ['blue', 'red', 'yellow', 'gray', 'green', 'yellow', 'pink', 'black', 'white', 'green']



fig = go.Figure(data=[go.Scatter(x=data.international, y=data.income,

                                 mode="markers",

                                 marker=dict(size=data.income, color= colors, line=dict(color='#000000', width=0.5)),

                                )

      ])

fig.show()
# data preparation | datani tayyorlash

df2016 = timesData[timesData.year == 2016].iloc[:20, :]

num_students = [float(each.replace(',', '.')) for each in df2016.num_students]

colors = [float(each.replace(',', '.')) for each in df2016.international]



# taking abbreviation of University name | Universitet nomini qisqartmasini olish

university_name = []

for names in df2016.university_name:

    abb = ''

    for name in names.split():

        if name[0].isupper():

            abb = abb + name[0]

    university_name.append(abb)



# vizualization

annotations = []

for u_name, rank, teaching, size in zip(university_name, df2016.world_rank, df2016.teaching, num_students):

    annotations.append({

            "font": { "size": 10 },

            "showarrow": False,

            "text": u_name,

            "x": float(rank),

            "y": float(teaching) + len(str(round(size * 2)))

    })

    

fig = {

    'data': [{

        'x': df2016.world_rank,

        'y': df2016.teaching,

        'mode': 'markers',

        'marker': {

            'size': num_students,

            'color': colors,

            'showscale': True

        },

        'text': df2016.university_name

    }],

    'layout': {

        'title': """Top 20 Universities World Rank and Teaching in 2016<br>

2016-yil top 20 universitet World Rank va Teachingi""",

        'annotations': annotations,

        'xaxis': {'title': 'World Rank' },

        'yaxis': {'title': 'Teaching' }

    }

}

iplot(fig)
# data preparation | datani tayyorlash

x2011 = timesData.student_staff_ratio[timesData.year == 2011]

x2012 = timesData.student_staff_ratio[timesData.year == 2012]



# vizualization

fig = {

    'data': [{

        'x': x2011,

        'opacity': .75,

        'name': '2011',

        'type': 'histogram',

        'marker': {

            'color': 'rgba(171, 50, 96, 0.6)',

        }

    }, {

        'x': x2012,

        'opacity': .75,

        'name': '2012',

        'type': 'histogram',

        'marker': {

            'color': 'rgba(12, 50, 196, 0.6)',

        }

    }],

    'layout': {

        'barmode': 'overlay',

        'title': 'Student Staff Ratio in 2011 and 2012',

        'xaxis': {'title': 'Student Staff Ratio', 'zeroline': True, 'zerolinewidth': 3, 'zerolinecolor': 'DarkGreen' },

        'yaxis': {'title': 'Count', 'zeroline': True, 'zerolinewidth': 3, 'zerolinecolor': 'green' }

    }

}



iplot(fig)
# Let's see what we have | Nimalar bor ekan?

timesData.country.value_counts()
# data preparation

countries = timesData.country # [timesData.year == 2011]

countries.replace(['United States of America'], 'USA', inplace = True)

countries.replace(['United Kingdom'], 'UK', inplace = True)

# vizualization

plt.subplots(figsize=(8, 8))



wordCloud = WordCloud(

    background_color = 'white',

    width = 512,

    height = 384,

    ).generate(" ".join(countries))



plt.imshow(wordCloud)

plt.axis('off')

plt.savefig('my_interesting_graph.png')



plt.show()
# data preparation | datani tayyorlash

x2015 = timesData[timesData.year == 2015]



# vizualization

fig = {

    'data': [{

        'y': x2015.total_score,

        'opacity': .75,

        'name': 'Total Score of Universities in 2015',

        'type': 'box',

        'marker': {

            'color': 'rgb(12, 12, 140)',

        }

    }, {

        'y': x2015.research,

        'opacity': .75,

        'name': 'Research of Universities in 2015',

        'type': 'box',

        'marker': {

            'color': 'rgb(12, 128, 128)',

        }

    }],

    'layout': {

        'boxmode': 'overlay',

        'title': """Total Score and Research of Universities in 2015<br>

2015 yilda universitetlarning umumiy bahosi va tadqiqotlari""",

    }

}



iplot(fig)
# import figure factory

import plotly.figure_factory as ff



# data preparation | datani tayyorlash

df = timesData[timesData.year == 2015]

df2015 = df.loc[:, ['research', 'international', 'total_score']]

df2015['index'] = np.arange(1, len(df2015) + 1)



# vizualization

fig = ff.create_scatterplotmatrix(df2015, diag = 'box', index = 'index', colormap = 'Portland',

                                 colormap_type = 'cat',

                                 height = 700, width = 700)

iplot(fig)
fig = {

    'data': [{

        'x': df.world_rank,

        'y': df.teaching,

        'name': 'Teaching',

        'marker': {

            'color': 'rgba(16, 112, 2, 0.8)'

        },

        'type': 'scatter',

        'mode': 'lines+markers',

    }, {

        'x': df.world_rank,

        'y': df.income,

        'xaxis': 'x2',

        'yaxis': 'y2',

        'name': 'Income',

        'marker': {

            'color': 'rgba(160, 112, 20, 0.8)'

        },

        'type': 'scatter',

        'mode': 'lines',

    },],

    'layout': {

        'xaxis2': {

            'domain': [.6, .95],

            'anchor': 'y2',

        },

        'yaxis2': {

            'domain': [.6, .95],

            'anchor': 'x2',

        },

        'title': 'Income and Teaching vs World Rank of Universities',

        'xaxis': { 'title': 'World Rank' },

    }

}

iplot(fig)
df.info()
# giving title to x, y, z axis

# version-1

fig = {

    'data': [{

        'x': [1, 2],

        'y': [1, 2],

        'z': [1, 2],

        'name': 'Legendary',

        'type': 'scatter3d',

        'mode': 'markers',

    },],

    'layout': {

        'showlegend': False,

        'scene':{

            'xaxis':{ 'title': 'x axis' },

            'yaxis':{ 'title': 'y axis' },

            'zaxis':{ 'title': 'z axis' },

        }

    }

}

iplot(fig)

# version-2



# trace1 = go.Scatter3d(

#     x=[1, 2],

#     y=[1, 2],

#     z=[1, 2],

#     name='Legendary'

# )

# data = go.Data([trace1])

# layout = go.Layout(

#     showlegend=True,

#     scene=go.Scene(

#         xaxis=dict(title='x axis title'),

#         yaxis=dict(title='y axis title'),

#         zaxis=dict(title='z axis title')

#     )

# )



# go.FigureWidget(data=data, layout=layout)
# data preparation | datani tayyorlash

hover_info = []

for name, rank, research, citation in zip(df.university_name, df.world_rank, df.research, df.citations):

    hover_info.append("Name: {}<br>World Rank: {}<br>Research: {}<br>Citations: {}".format(name, rank, research, citation))



# vizualization

fig = {

    'data':[{

        'x': df.world_rank,

        'y': df.research,

        'z': df.citations,

        'text':  hover_info,

        'hoverinfo': 'text', # 'skip', 'text', default: 'x+y+z'

        'type': 'scatter3d',

        'mode': 'lines+markers',

        'marker': {

            'size': 10,

            'color': 'blue',

        }

    }],

    'layout': {

        'title': '3D plot World Rank, Research and Citations of universities in 2015',

        'margin': {

            'l': 0,

            'r': 0,

            't': 0,

            'b': 0,

        },

        'scene': {

            'xaxis': { 'title': 'World Rank' },

            'yaxis': { 'title': 'Research' },

            'zaxis': { 'title': 'Citations' },

        }

        

    }

}

iplot(fig)
# making tarces | tracelar hosil qilish

trace1 = go.Scatter(

    x = df.world_rank,

    y = df.research,

    name = 'Research',

)

trace2 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Citations',

)

trace3 = go.Scatter(

    x = df.world_rank,

    y = df.income,

    xaxis = 'x3',

    yaxis = 'y3',

    name = 'Income',

)

trace4 = go.Scatter(

    x = df.world_rank,

    y = df.total_score,

    xaxis = 'x4',

    yaxis = 'y4',

    name = 'Total Score',

)



# making figure

data = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis = { 'domain': [0, .45], },

    yaxis = { 'domain': [0, .45], },

    

    xaxis2 = { 'domain': [.55, 1], 'anchor': 'y2' },

    yaxis2 = { 'domain': [0, .45], 'anchor': 'x2' },

    

    xaxis3 = { 'domain': [0, .45], 'anchor': 'y3' },

    yaxis3 = { 'domain': [.55, 1], 'anchor': 'x3' },

    

    xaxis4 = { 'domain': [.55, 1], 'anchor': 'y4' },

    yaxis4 = { 'domain': [.55, 1], 'anchor': 'x4' },

    

    title = 'Research, Citation, Income and Total Score vs World Rank of Universities'

)



# vizualization

fig = go.Figure(data = data, layout = layout)

iplot(fig)