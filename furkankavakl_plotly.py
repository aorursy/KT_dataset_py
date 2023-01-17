import numpy as np # linear algebra

import pandas as pd # data processing

#plotly

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

#word cloud

from wordcloud import WordCloud

#matplotlib

import matplotlib.pyplot as plt

import plotly.graph_objs as go



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
timesData = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')
timesData.head()
timesData[timesData.university_name.str.contains("California Institute of Technology")]
timesData.iloc[1803:1804,:]
df=timesData.iloc[:100,:]



import plotly.graph_objs as go



trace1=go.Scatter(

                  x=df.world_rank,

                  y=df.citations,

                  mode='lines+markers',

                  name='citations',

                  marker=dict(color='rgba(16,112,2,0.8)'),

                  text=df.university_name)



trace2=go.Scatter(

                  x=df.world_rank,

                  y=df.teaching,

                  mode='lines+markers',

                  name='teaching',

                  marker=dict(color="rgba(80,26,80,0.8)"),

                  text=df.university_name)



data=[trace1,trace2]



layout=dict(title='Citaiton and Teaching vs World Rank of Top 100 Universities',

            xaxis=dict(title='World Rank',ticklen=5,zeroline=False))



fig=dict(data=data,layout=layout)

iplot(fig)
df14=timesData[timesData.year == 2014].iloc[:100,:]

df15=timesData[timesData.year == 2015].iloc[:100,:]

df16=timesData[timesData.year == 2016].iloc[:100,:]



trace1 = go.Scatter(

                    x=df14.world_rank,

                    y=df14.citations,

                    mode='markers',

                    name='2014',

                    marker=dict(color='rgba(255,128,255,0.8)'),

                    text = df14.university_name)



trace2 = go.Scatter(

                    x=df15.world_rank,

                    y=df15.citations,

                    mode='markers',

                    name='2015',

                    marker=dict(color='rgba(255,128,2,0.8)'),

                    text = df15.university_name)



trace3 = go.Scatter(

                    x=df16.world_rank,

                    y=df16.citations,

                    mode='markers',

                    name='2016',

                    marker=dict(color='rgba(0,255,200,0.8)'),

                    text = df16.university_name)



data = [trace1,trace2,trace3]

layout=dict(title='Citation vs World Rank of Top 100 Universities in the years of 2014, 2015, 2016',

            xaxis=dict(title="World Rank",ticklen=5,zeroline=False),

            yaxis=dict(title="Citation",ticklen=5,zeroline=False))



fig=dict(data=data,layout=layout)

iplot(fig)
df14=timesData[timesData.year==2014].iloc[:3,:]



trace1=go.Bar(

                x=df14.university_name,

                y=df14.citations,

                name='Citations',

                marker=dict(color='rgba(255,174,255,0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

                text=df14.country)



trace2=go.Bar(

                x=df14.university_name,

                y=df14.teaching,

                name='Teaching',

                marker=dict(color='rgba(255,255,128,0.5)',line=dict(color='rgb(0,0,0)',width=1.5)),

                text=df14.country)



data=[trace1,trace2]

layout=go.Layout(barmode='group')

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df14=timesData[timesData.year==2014].iloc[:3,:]



x=df14.university_name



trace1={

    'x' : x,

    'y' : df14.citations,

    'name' : 'citation',

    'type' : 'bar'

};



trace2={

    'x':x,

    'y':df14.teaching,

    'name':'teaching',

    'type':'bar'

};



data=[trace1,trace2]

layout={

    'xaxis':{'title':'Top 3 Universities'},

    'barmode':'relative',

    'title':'citations and teching of top 3 universities'

};

fig=go.Figure(data=data,layout=layout)

iplot(fig)
df16.info()
df16=timesData[timesData.year==2016].iloc[:7,:]

pie1=df16.num_students

pie1_list=[float(each.replace(',','.')) for each in df16.num_students]

labels=df16.university_name



fig={

    'data':[

        {

            'values':pie1_list,

            'labels':labels,

            'domain':{'x':[0,.5]},

            'name':'Number of Students Rates',

            'hoverinfo':'label+percent+name',

            'hole':.3,

            'type':'pie'

        },

    ],

    'layout':{

        'title':'Universities Number of Student Rates',

        'annotations':[

            {'font':{'size':20},

             'showarrow':False,

             'text':'Number of Students',

             'x':0.13,

             'y':1.1},

        ]

    }

}

iplot(fig)
df16=timesData[timesData.year == 2016].iloc[:20,:]

num_students_size=[float(each.replace(',','.'))for each in df16.num_students]

international_color=[float(each)for each in df16.international]

data=[

    {

        'y':df16.teaching,

        'x':df16.world_rank,

        'mode':'markers',

        'marker':{

            'color':international_color,

            'size':num_students_size,

            'showscale':True

        },

        'text':df16.university_name

    }

]

iplot(data)
x11=timesData.student_staff_ratio[timesData.year==2011]

x12=timesData.student_staff_ratio[timesData.year==2012]



trace1=go.Histogram(

                    x=x11,

                    opacity=0.75,

                    name='2011',

                    marker=dict(color='rgba(171,50,96,0.6)'))

trace2=go.Histogram(

                    x=x12,

                    opacity=0.75,

                    name='2012',

                    marker=dict(color='rgba(12,50,196,0.6)'))



data=[trace1,trace2]

layout=go.Layout(barmode='overlay',

                 title='student-staff ratio in 2011/12',

                 xaxis=dict(title='student-staff ratio'),

                yaxis=dict(title='Count'),)

fig=go.Figure(data=data,layout=layout)

iplot(fig)
x2011=timesData.country[timesData.year==2011]

plt.subplots(figsize=(8,8))

wordcloud=WordCloud(

                    background_color='white',

                    width=512,

                    height=384,

                    ).generate(' '.join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')

plt.show()
x2015=timesData[timesData.year==2015]



trace0=go.Box(

    y=x2015.total_score,

    name='total score of universities in 2015',

    marker=dict(color='rgb(12,12,140)',)

)



trace1=go.Box(

    y=x2015.research,

    name='research of university in 2015',

    marker=dict(color='rgb(12,128,128)',)

)

data=[trace0,trace1]

iplot(data)
import plotly.figure_factory as ff

dataframe=timesData[timesData.year==2015]

data2015=dataframe.loc[:,['research','international','total_score']]

data2015['index']=np.arange(1,len(data2015)+1)

fig=ff.create_scatterplotmatrix(data2015,diag='box',index='index',colormap='Portland',colormap_type='cat',height=700,width=700)

iplot(fig)
trace1=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name='teaching',

    marker=dict(color='rgba(16,112,2,0.8)',)

)

trace2=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',

    yaxis='y2',

    name='income',

    marker=dict(color='rgba(160,112,20,0.8)',)

)

data=[trace1,trace2]

layout=go.Layout(

    xaxis2=dict(domain=[0.6,0.95],anchor='y2'),

    yaxis2=dict(domain=[0.6,0.95],anchor='x2'),

    title='Income and Teaching vs World Rank of Universities')

fig=go.Figure(data=data,layout=layout)

iplot(fig)
trace1=go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(size=10,color='rgb(255,0,0)'))



data=[trace1]



layout=go.Layout(margin=dict(l=0,r=0,b=0,t=0))

fig=go.Figure(data=data,layout=layout)

iplot(fig)
trace1=go.Scatter(x=dataframe.world_rank,y=dataframe.research,name='research')

trace2=go.Scatter(x=dataframe.world_rank,y=dataframe.citations,xaxis='x2',yaxis='y2',name='citations')

trace3=go.Scatter(x=dataframe.world_rank,y=dataframe.income,xaxis='x3',yaxis='y3',name='income')

trace4=go.Scatter(x=dataframe.world_rank,y=dataframe.total_score, xaxis='x4',yaxis='y4',name='total_score')

data=[trace1,trace2,trace3,trace4]

layout=go.Layout(xaxis=dict(domain=[0, 0.45]),

                 yaxis=dict(domain=[0,0.45]),

                 xaxis2=dict(domain=[0.55,1]),

                 yaxis2=dict(domain=[0,0.45],anchor='x2'),    

                 xaxis3=dict(domain=[0,0.45],anchor='y3'),    

                 yaxis3=dict(domain=[0.55,1]),    

                 xaxis4=dict(domain=[0.55,1],anchor='y4'),    

                 yaxis4=dict(domain=[0.55,1],anchor='x4'),

                 title= 'Research, citation, income and total score VS World Rank of Universities')

fig=go.Figure(data=data,layout=layout)

iplot(fig)