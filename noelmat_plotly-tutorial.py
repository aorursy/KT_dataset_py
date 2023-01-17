# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import plotly.graph_objects as go



from wordcloud import WordCloud



import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
timesData = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')
timesData.info()
timesData.head(10)
#prepare data frame

df = timesData.iloc[:100,:]



trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = 'lines',

                    name = 'citations',

                    marker = dict(color='rgba(16,112,2,0.8)'),

                    text = df.university_name

)



trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = 'lines+markers',

                    name = 'teaching',

                    marker = dict(color = 'rgba(80,26,80,0.8)'),

                    text = df.university_name

)



data= [trace1,trace2]



layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

             xaxis = dict(title= 'World Rank', ticklen=5, zeroline=False)

             )

fig = dict(data=data, layout=layout)

go.Figure(fig).show()



# #import graph objects as "go"

# import plotly.graph_objects as go



# # Creating trace1

# trace1 = go.Scatter(

#                     x = df.world_rank,

#                     y = df.citations,

#                     mode = 'lines',

#                     name = 'citations',

#                     marker= dict(color = 'rgba(16,112,2,0.8)'),

#                     text = df.university_name)



# # Creating trace2

# trace2 = go.Scatter(

#                     x= df.world_rank,

#                     y= df.teaching,

#                     mode= 'lines+markers',

#                     name = 'teaching',

#                     marker = dict(color = 'rgba(80,26,80,0.8)'),

#                     text = df.university_name)



# data = [trace1, trace2]

# layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

#              xaxis= dict(title='World Rank', ticklen=5, zeroline=False)

#              )



# fig = dict(data= data,layout=layout)

# iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]



trace1 = go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = 'markers',

                    name = '2014',

                    marker = dict(color = 'rgba(255,128,255,0.8)'),

                    text = df2014.university_name)



trace2 = go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = 'markers',

                    name = '2015',

                    marker = dict(color = 'rgba(255,128,2,0.8)'),

                    text = df2015.university_name)



trace3 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = 'markers',

                    name = '2016',

                    marker = dict(color='rgba(0,255,200,0.8)'),

                    text = df2016.university_name)

data = [trace1, trace2, trace3]

layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',

             xaxis = dict(title= 'World Rank',ticklen=5, zeroline=False))

fig = dict(data=data, layout= layout)

go.Figure(fig).show()
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

df2014
#prepare data frames

df2014 = timesData[timesData.year== 2014].iloc[:3,:]

#create trace1

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = 'citations',

                marker = dict(color='rgba(255,174,255,0.5)',

                             line = dict(color = 'rgb(0,0,0)',width=1.5)),

                text = df2014.country)

#create trace2

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = 'teaching',

                marker= dict(color= 'rgba(255,255, 128,0.5)',

                            line = dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)

data= [trace1,trace2]

layout = go.Layout(barmode='group')

fig = go.Figure(data = data, layout=layout)

fig.show()

df2014 = timesData[timesData.year == 2014].iloc[:3,:]



x = df2014.university_name



trace1={

    'x': x,

    'y': df2014.citations,

    'name':'citation',

    'type': 'bar'

};



trace2={

    'x': x,

    'y': df2014.teaching,

    'name' : 'teaching',

    'type' : 'bar'

};



data = [trace1, trace2];

layout = {

    'xaxis': {'title': 'Top 3 universities'},

    'barmode': 'relative',

    'title' : 'citations and teaching of top 3 universities in 2014'

};



fig = go.Figure(data = data, layout=layout)

fig.show()
from plotly import subplots



df2016 = timesData[timesData.year == 2016].iloc[:7,:]



y_saving = [each for each in df2016.research]

y_net_worth = [float(each) for each in df2016.income]

x_saving = [each for each in df2016.university_name]

x_net_worth = [each for each in df2016.university_name]



trace0 = go.Bar(

                x = y_saving,

                y = x_saving,

                marker = dict(color='rgba(171, 50 ,96, 0.6)',

                             line= dict(color='rgba(171,50,96,1.0)', width=1)),

                name='research',

                orientation = 'h',

)



trace1 = go.Scatter(

                    x = y_net_worth,

                    y = x_net_worth,

                    mode = 'lines+markers',

                    line=dict(color='rgb(63,72,204)'),

                    name = 'income',

)



layout = dict(

                title='Citations and income',

                yaxis=dict(showticklabels=True,domain=[0,0.85]),

                yaxis2 = dict(showline=True,showticklabels=False, linecolor='rgba(102,102,102,0.8)',linewidth=2,domain=[0,0.85]),

                xaxis = dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0,0.42]),

                xaxis2 = dict(zeroline=False,showline=False,showticklabels=True,showgrid=True,domain=[0.47,1],side='top',dtick=25),

                legend = dict(x=0.029,y=1.038,font=dict(size=10)),

                margin=dict(l=200,r=20,t=70,b=70),

                paper_bgcolor='rgb(248, 248, 255)',

                plot_bgcolor='rgb(248,248,255)',

)



annotations = []

y_s = np.round(y_saving, decimals=2)

y_nw =np.rint(y_net_worth)



for ydn, yd, xd in zip(y_nw, y_s, x_saving):

    annotations.append(dict(xref='x2',yref='y2',y = xd, x=ydn -4 , text='{:,}'.format(ydn),font=dict(family='Arial',size=12,color='rgb(171,50,96)'),showarrow=False))

    annotations.append(dict(xref='x1',yref='y1', y=xd, x=yd + 3, text=str(yd), font=dict(family='Arial', size=12, color= 'rgb(171,50,96)'),showarrow=False))

layout['annotations'] = annotations



fig = subplots.make_subplots(rows=1, cols=2, specs=[[{},{}]],shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(trace0,1,1)

fig.append_trace(trace1,1,2)



fig['layout'].update(layout)

fig.show()
df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',','.')) for each in df2016.num_students]

labels = df2016.university_name



fig = {

    'data': [

        {

            'values': pie1_list,

            'labels': labels,

            'domain': { 'x': [0, .5]},

            'name': 'Number Of Students Rates',

            'hoverinfo': 'label+percent+name',

            'hole': .5,

            'type': 'pie'

        },],

    'layout': {

        'title': 'Universities Number of Students rates',

        'annotations': [

            {'font': {'size':20},

             'showarrow':False,

             'text': 'Number of Students',

             'x': 0.20,

             'y': 1

            },

        ]

    }

}

go.Figure(fig).show()
df2016.info()
df2016 = timesData[timesData.year == 2016].iloc[:20,:]

num_students_size = [float(each.replace(',','.')) for each in df2016.num_students]

international_color = [float(each) for each in df2016.international]

data = [

    {

        'y': df2016.teaching,

        'x': df2016.world_rank,

        'mode': 'markers',

        'marker': {

            'color': international_color,

            'size' : num_students_size,

            'showscale' : True

        },

        'text': df2016.university_name

    }

]

go.Figure(data).show()
x2011 = timesData.student_staff_ratio[timesData.year == 2011]

x2012 = timesData.student_staff_ratio[timesData.year == 2012]



trace1 = go.Histogram(

        x= x2011,

        opacity = 0.75,

        name = '2011',

        marker = dict(color = 'rgba(171,50,96,0.6)'))



trace2 = go.Histogram(

        x = x2012,

        opacity=0.75,

        name = '2012',

        marker = dict(color = 'rgba(12,50,196,0.6)'))



data = [trace1,trace2]

layout = go.Layout(barmode='overlay',

                   title='student-staff ratio in 2011 and 2012',

                   xaxis=dict(title='students-staff ratio'),

                   yaxis=dict(title='Count'),)



fig = go.Figure(data=data, layout=layout)

fig.show()
x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                        background_color='white',

                        width=512,

                        height=384

                        ).generate(' '.join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
x2015 = timesData[timesData.year == 2015]



trace0 = go.Box(

    y=x2015.total_score,

    name = 'total score of universities in 2015',

    marker = dict(

        color = 'rgb(12,12,140)',

    )

)



trace1 = go.Box(

    y=x2015.research,

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(12,128,128)',

    )

)

data = [trace0,trace1]

go.Figure(data).show()
import plotly.figure_factory as ff



dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,['research','international','total_score']]

data2015['index'] = np.arange(1,len(data2015)+1)



fig = ff.create_scatterplotmatrix(data2015,diag = 'box', index ='index',colormap='Portland', colormap_type = 'cat',

                                 height = 700, width=700)

fig.show()
trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name='teaching',

    marker=dict(color='rgba(16,112,2,0.8)'),

)



trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',

    yaxis='y2',

    name='income',

    marker=dict(color='rgba(160,112,20,0.8)'),

)

data = [trace1,trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.5,0.95],

        anchor='y2',

    ),

    yaxis2=dict(

        domain=[0.6,0.95],

        anchor='x2',

    ),

    title = 'Income and Teaching vs World Rank of Universities'

    

)



fig = go.Figure(data=data,layout=layout)

fig.show()
trace1 = go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',

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



fig = go.Figure(data=data,layout=layout)

fig.show()