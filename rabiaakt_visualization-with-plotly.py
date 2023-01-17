# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go



from wordcloud import WordCloud

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
times_data = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')

times_data.head()
times_data.info()
times_data.female_male_ratio.unique()
df = times_data.iloc[:100, :]



trace1 = go.Scatter(

    x = df.world_rank,

    y = df.citations,

    mode = "lines",

    name = "citations",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

    text= df.university_name)



trace2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    mode = "lines+markers",

    name = "teaching",

    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

    text= df.university_name)



dat = [trace1, trace2]

layout = dict(title = 'Citiation and Teaching vs World Rank Top 100 University', 

              xaxis = dict(title = 'World Rank', ticklen = 7, zeroline = False))

fig = dict(data = dat, layout = layout)

iplot(fig)
df2014 = times_data[times_data.year == 2014].iloc[:100, :]

df2015 = times_data[times_data.year == 2015].iloc[:100, :]

df2016 = times_data[times_data.year == 2016].iloc[:100, :]



trace1 = go.Scatter(

    x = df2014.world_rank,

    y = df2014.total_score,

    mode = "markers",

    name = "2014",

    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

    text = df2014.university_name)



trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.total_score,

    mode = 'markers',

    name = '2015',

    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

    text = df2015.university_name)



trace3 = go.Scatter(

    x = df2016.world_rank,

    y = df2016.total_score,

    mode = 'markers',

    name = '2016',

    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

    text = df2016.university_name)



data = [trace1, trace2, trace3]

layout = dict(title = 'Citiation point according to years 2014, 2015, 2016',

             xaxis = dict(title = "World Rank", ticklen = 5, zeroline = True))

fig = dict(data = data, layout = layout)

iplot(fig)
df2014 = times_data[times_data.year == 2014].iloc[:100, :]

df2015 = times_data[times_data.year == 2015].iloc[:100, :]

df2016 = times_data[times_data.year == 2016].iloc[:100, :]



trace1 = go.Scatter(

    x = df2014.world_rank,

    y = df2014.citations,

    mode = "markers",

    name = "2014",

    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

    text = df2014.university_name)



trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.citations,

    mode = 'markers',

    name = '2015',

    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

    text = df2015.university_name)



trace3 = go.Scatter(

    x = df2016.world_rank,

    y = df2016.citations,

    mode = 'markers',

    name = '2016',

    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

    text = df2016.university_name)



data = [trace1, trace2, trace3]

layout = dict(title = 'Citiation point according to years 2014, 2015, 2016',

             xaxis = dict(title = "World Rank", ticklen = 5, zeroline = True))

fig = dict(data = data, layout = layout)

iplot(fig)
top3 = df2014.iloc[:3, :]



trace = go.Bar(

    x = top3.university_name,

    y = top3.citations,

    name = "citations",

    marker = dict(color = 'rgba(255, 174, 255, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

    text = top3.country

)



trace2 = go.Bar(

    x = top3.university_name,

    y = top3.teaching,

    name = "teaching",

    marker = dict(color = 'rgba(255, 255, 128, 0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

    text = top3.country

)

data = [trace, trace2]

layout = go.Layout(barmode = 'group')

fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace = {

    'x' : top3.university_name,

    'y' : top3.citations,

    'name' : 'citations',

    'type' : 'bar'

};

trace1 = {

    'x' : top3.university_name,

    'y' : top3.teaching,

    'name' : 'teaching',

    'type' : 'bar'

};

data = [trace, trace1]

layout = {

    'xaxis' : {'title' : 'Top 3 universities'},

    'barmode' : 'relative',

    'title': 'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)
df2016 = times_data[times_data.year == 2016].iloc[:7, :]

df2016.info()
pie1 = df2016.num_students

pie_list = [float(each.replace(',','.')) for each in df2016.num_students]

labels = df2016.university_name



fig = {

    "data" : [

        {

            "values" : pie_list,

            "labels" : labels,

            "domain" : {"x" : [0, .5]},

            "name": "Number Of Students Rates",

            "hoverinfo":"label+percent+name",

            "hole": .3,

            "type": "pie"

        },

    ],

    "layout" : {

          "title":"Universities Number of Students rates",

          "annotations": [

            { 

                "font": { "size": 20},

                "showarrow": False,

                "text": "Number of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
df2016  = times_data[times_data.year == 2016].iloc[:20, :]

numOfStd = [float(each.replace(',','.')) for each in df2016.num_students]

intColor = [float(each) for each in df2016.international]

#intColor = df2016.international

data = [

    {

        'y' : df2016.teaching,

        'x' : df2016.world_rank,

        'mode' : 'markers',

        'marker' : {

            'color' : intColor,

            'size' : numOfStd,

            'showscale' : True

        },

        'text' : df2016.university_name

    }

]



iplot(data)
ss2011 = times_data.student_staff_ratio[times_data.year == 2011]

ss2012 = times_data.student_staff_ratio[times_data.year == 2012]



trace = go.Histogram(

    x = ss2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)')

)



trace2 = go.Histogram(

    x = ss2012,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)')

)

data = [trace, trace2]

layout = go.Layout(barmode='overlay',

                   title=' students-staff ratio in 2011 and 2012',

                   xaxis=dict(title='students-staff ratio'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
y2011 = times_data.country[times_data.year == 2011]

plt.subplots(figsize = (9,9))



wordcloud = WordCloud(

            background_color = 'white',

            width=512,

            height=384

            ).generate(" ".join(y2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('word.png')



plt.show()

y2015 = times_data[times_data.year == 2015]



trace = go.Box(

        y = y2015.total_score,

        name = 'Total score of universities in 2015',

        marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)



trace2 = go.Box(

        y = y2015.research,

        name = 'Research score of universities in 2015',

        marker = dict(

            color = "rgb(255, 14, 100)"

        )

)



data = [trace, trace2]

iplot(data)
import plotly.figure_factory as ff



y2015 = times_data[times_data.year == 2015]

data2015 = y2015.loc[:,["research","international", "total_score"]] 

data2015["index"] = np.arange(1,len(data2015)+1)



figure = ff.create_scatterplotmatrix(data2015, diag = 'box', index = 'index', colormap = 'Portland', colormap_type = 'cat',  height=700, width=700)

iplot(figure)
trace = go.Scatter(

    x = times_data.world_rank,

    y = times_data.teaching,

    name = 'Teaching score',

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)



trace2 = go.Scatter(

    x = times_data.world_rank,

    y = times_data.income,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Income',

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

data =  [trace, trace2]

layout = go.Layout(

     xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor= 'x2',

    ),

    title = 'Income and Teaching vs World Rank of Universities'

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)
trace1  = go.Scatter3d(

    x = times_data.world_rank,

    y = times_data.research,

    z = times_data.citations,

    mode = 'markers',

    marker = dict(

        size = 7,

        color = 'rgba(170, 40, 45)'

        #colorscale = 'rocket'

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

    x = times_data.world_rank,

    y = times_data.research,

    name = 'Research'

)



trace2 = go.Scatter(

    x = times_data.world_rank,

    y = times_data.citations,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'Citations'

)



trace3 = go.Scatter(

    x = times_data.world_rank,

    y = times_data.income,

    xaxis = 'x3',

    yaxis = 'y3',

    name = 'Income'

)



trace4 = go.Scatter(

    x = times_data.world_rank,

    y = times_data.total_score,

    xaxis = 'x4',

    yaxis = 'y4',

    name = 'Total Score'

)



data = [trace1, trace2, trace3, trace4]



layout = go.Layout(

    xaxis = dict(

        domain = [0, 0.45]

    ),

    yaxis = dict(

        domain = [0, 0.45]

    ),

    xaxis2 = dict(

        domain = [0.55 , 1]

    ),

    yaxis2 = dict(

        domain = [0, 0.45],

        anchor = 'x2'

    ),

    xaxis3 = dict(

        domain = [0 , 0.45],

        anchor = 'y3'

    ),

    yaxis3 = dict(

        domain = [0.55, 1],

        anchor = 'x3'

    ),

    xaxis4 = dict(

        domain = [0.55 , 1],

        anchor = 'y4'

    ),

    yaxis4 = dict(

        domain = [0.55, 1],

        anchor = 'x4'

    ),

    

)



fig = go.Figure(data = data, layout = layout)

iplot(fig)