# data analysis

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import matplotlib.pyplot as plt



# plotly library

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



import os

print(os.listdir("../input"))
tdata = pd.read_csv('../input/timesData.csv')
tdata.info()
tdata.head()
# Line Charts Example: Citation and Teaching vs World Rank of Top 100 Universities

# Preparing data frame

df = tdata.iloc[:100,:]



# graph_objs library is needed for plotting

import plotly.graph_objs as go



# Creating traces for representing the specified features

trace1 = go.Scatter (

    x = df.world_rank,

    y = df.citations,

    mode = "lines",

    name = "citations",

    marker = dict( color = 'rgba(0, 0, 255, 0.8)'), # rgba represents red, green, blue, alpha(opacity)

    text = df.university_name)



trace2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    mode = "lines+markers",

    name = "teaching",

    marker = dict( color = 'rgba(255, 0, 0, 0.8)'),

    text = df.university_name)



data = [trace1,trace2]

layout = dict(title = 'Citation and Teaching - World Rank of Top 100 Universities',

              xaxis = dict( title = 'World Rank', ticklen = 5, zeroline = False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# Scatter Example: Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years



df2014 = tdata[tdata.year == 2014].iloc[:100,:]

df2015 = tdata[tdata.year == 2015].iloc[:100,:]

df2016 = tdata[tdata.year == 2016].iloc[:100,:]



import plotly.graph_objs as go



trace1 = go.Scatter(

    x = df2014.world_rank,

    y = df2014.citations,

    mode = "markers",

    name = "2014",

    marker = dict ( color = 'rgba( 0, 153, 204, 0.4)'),

    text = df2014.university_name    

    )



trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.citations,

    mode = "markers",

    name = "2015",

    marker = dict ( color = 'rgba( 255, 51, 153, 0.6)'),

    text = df2015.university_name    

    )



trace3 = go.Scatter(

    x = df2016.world_rank,

    y = df2016.citations,

    mode = "markers",

    name = "2016",

    marker = dict ( color = 'rgba( 153, 255, 204, 0.8)'),

    text = df2016.university_name    

    )



data = [trace1, trace2, trace3]

layout = dict(title ='Citation - World Rank of Top 100 Universities with 2014, 2015 and 2016 years', 

              xaxis = dict(title ='World Rank', ticklen = 5, zeroline = False),

              yaxis = dict(title ='Citations', ticklen = 5, zeroline = False))

fig = dict( data = data, layout = layout)

iplot(fig)
# Scatter : Total Score vs world rank of top 100 universities with 2014, 2015 and 2016 years



df2014 = tdata[tdata.year == 2014].iloc[:100,:]

df2015 = tdata[tdata.year == 2015].iloc[:100,:]

df2016 = tdata[tdata.year == 2016].iloc[:100,:]



import plotly.graph_objs as go



trace1 = go.Scatter(

    x = df2014.world_rank,

    y = df2014.total_score,

    mode = "markers",

    name = "2014",

    marker = dict ( color = 'rgba( 0, 153, 204, 0.4)'),

    text = df2014.university_name    

    )



trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.total_score,

    mode = "markers",

    name = "2015",

    marker = dict ( color = 'rgba( 255, 51, 153, 0.6)'),

    text = df2015.university_name    

    )



trace3 = go.Scatter(

    x = df2016.world_rank,

    y = df2016.total_score,

    mode = "markers",

    name = "2016",

    marker = dict ( color = 'rgba( 153, 255, 204, 0.8)'),

    text = df2016.university_name    

    )



data = [trace1, trace2, trace3]

layout = dict(title ='Total Score - World Rank of Top 100 Universities with 2014, 2015 and 2016 years', 

              xaxis = dict(title ='World Rank', ticklen = 5, zeroline = False),

              yaxis = dict(title ='Total Score', ticklen = 5, zeroline = False))

fig = dict( data = data, layout = layout)

iplot(fig)
df2014 = tdata[tdata.year == 2014].iloc[:3,:] # This is a data frame that will be plot bar chart.

df2014
# First Bar Charts Example: citations and teaching of top 3 universities in 2014 (style1)

df2014 = tdata[tdata.year == 2014].iloc[:3,:]

# Importing library

import plotly.graph_objs as go

# Creating traces

trace1 = go.Bar(

    x = df2014.university_name,

    y = df2014.citations,

    name = "citations",

    marker = dict( color = 'rgba( 51, 102, 255, .8)', 

                  line = dict( color = 'rgb( 0, 0, 0)', width = 1.5)),

    text = df2014.country    

)



trace2 = go.Bar(

    x = df2014.university_name,

    y = df2014.teaching,

    name = "teaching",

    marker = dict( color = 'rgba( 255, 204, 102, .8)', 

                  line = dict( color = 'rgb( 0, 0, 0)', width = 1.5)),

    text = df2014.country

)

# Plotting processes

data = [trace1, trace2]



layout = go.Layout( barmode = "group")



fig = go.Figure(data = data, layout = layout)

iplot(fig)
# Second Bar Charts Example: citations and teaching of top 3 universities in 2014 (style2) 

df2014 = tdata[tdata.year == 2014].iloc[:3,:]

# Importing library

import plotly.graph_objs as go

# Creating traces







x = df2014.university_name



trace1 = {

    

    'x' : x,

    'y' :df2014.citations,

    'name' : "citations",

    'type' : 'bar'

}



trace2 = {

    

    'x' : x,

    'y' :df2014.teaching,

    'name' : "teaching",

    'type' : 'bar'

}

# Plotting processes

data = [trace1, trace2]



layout = {

  'xaxis': {'title': 'Top 3 Universities'},

  'barmode': 'relative',

  'title': 'Citations - Teaching of Top 3 Universities in 2014'

};



fig = go.Figure(data = data, layout = layout)

iplot(fig)
# import graph objects as "go" and import tools

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt

# prepare data frames

df2016 = tdata[tdata.year == 2016].iloc[:7,:]



y_saving = [each for each in df2016.research]

y_net_worth  = [float(each) for each in df2016.income]

x_saving = [each for each in df2016.university_name]

x_net_worth  = [each for each in df2016.university_name]

trace0 = go.Bar(

                x=y_saving,

                y=x_saving,

                marker=dict(color='rgba(171, 50, 96, 0.6)',line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

                name='research',

                orientation='h',

)

trace1 = go.Scatter(

                x=y_net_worth,

                y=x_net_worth,

                mode='lines+markers',

                line=dict(color='rgb(63, 72, 204)'),

                name='income',

)

layout = dict(

                title='Citations - Income',

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
# Pie Charts Example: Students rate of top 7 universities in 2016

df2016 = tdata[tdata.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]

labels = tdata.university_name



# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": 0,

      "type": "pie"

    },],

  "layout": {

        "title":"Universities Number of Students rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": True,

              "text": "Number of Students",

                "x": 0.1,

                "y": 1.1

            },

        ]

    }

}

iplot(fig)





# Bubble Charts Example: University world rank (first 20) vs teaching score with number of students(size) and

# international score (color) in 2016



# Data preparation

df2016 = tdata[tdata.year == 2016].iloc[:20,:]

num_students  = [float(each.replace(',', '.')) for each in df2016.num_students]

international = [float(each) for each in df2016.international]



# Chart preparation

data = [

    {

        'y' : df2016.teaching,

        'x' : df2016.world_rank,

        'mode' : 'markers',

        'marker': {

            'color': international,

            'size': num_students,

            'showscale': True

        },

        "text" : df2016.university_name

    }

]



iplot(data)
# Lets look at histogram of students-staff ratio in 2011 and 2012 years.



# Data preparation

ssr2011 = tdata.student_staff_ratio[tdata.year == 2011]

ssr2012 = tdata.student_staff_ratio[tdata.year == 2012]



# Chart preparation

trace1 = go.Histogram(

    x = ssr2011,

    opacity = 0.75,

    name = "2011",

    marker = dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x = ssr2012,

    opacity = 0.75,

    name = "2012",

    marker = dict(color='rgba(12, 50, 196, 0.6)'))



data = [ trace1, trace2]



layout = go.Layout( barmode= 'overlay',

                   title = ' students-staff ratio in 2011 and 2012',

                   xaxis = dict(title='students-staff ratio'),

                   yaxis = dict(title='Count'),

)



fig = go.Figure( data = data, layout = layout)

iplot(fig)
# Lets look at which country is mentioned most in 2011.

c2011 = tdata.country[tdata.year == 2011]

plt.subplots(figsize=(15,15))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(c2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
x2015 = tdata[tdata.year == 2015]



trace1 = go.Box(

    y = x2015.total_score,

    name = 'total score of universities in 2015',

    marker = dict(

        color = 'rgb(12, 12, 255)',

    )

)



trace2 = go.Box(

    y = x2015.research,

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(255, 153, 51)',

    )

)



data = [ trace1, trace2]

iplot(data)

import plotly.figure_factory as ff



# Preparation data

df2015 = tdata[tdata.year == 2015]

data2015 = df2015.loc[:,["research", "international", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)



# Scatter matrix

fig = ff.create_scatterplotmatrix( data2015, diag = 'box', index = 'index', colormap= 'Portland', 

                                  colormap_type = 'cat', height=700, width=700)

iplot(fig)
# First Plot

trace1 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.teaching,

    name = "teaching",

    marker = dict(color = 'rgba(0, 0, 255, 0.8)')

)

# Second Plot

trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.income,

    xaxis = 'x2',

    yaxis = 'y2',

    name = "income",

    marker = dict(color = 'rgba(255, 0, 0, 0.8)')

)



data = [ trace1, trace2]

layout = go.Layout( xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = 'Income and Teaching vs World Rank of Universities'

)



fig = go.Figure(data=data, layout=layout)

iplot(fig)







trace1 = go.Scatter3d(

    x = df2015.world_rank,

    y = df2015.research,

    z = df2015.citations,

    mode = 'markers',

    marker=dict(

        size=10,

        color='rgb(0,0,255)',# set color to an array/list of desired values

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

fig = go.Figure( data = data, layout = layout)

iplot(fig)
trace1 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.research,

    name = "research",

    marker = dict ( color = 'rgba(0,0,255,0.8)')

)



trace2 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.citations,

    xaxis = 'x2',

    yaxis = 'y2',

    name = "citations"

)



trace3 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.income,

    xaxis = 'x3',

    yaxis = 'y3',

    name = "income",

    marker = dict ( color = 'rgba(0,255,0,0.8)')

)



trace4 = go.Scatter(

    x = df2015.world_rank,

    y = df2015.total_score,

    xaxis = 'x4',

    yaxis = 'y4',

    name = "total_score",

    marker = dict ( color = 'rgba(255,0,0,0.8)')

)



data = [ trace1, trace2, trace3, trace4]



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

    title = 'Research, citation, income and total score VS World Rank of Universities'

)

fig = go.Figure( data = data, layout = layout)

iplot(fig)