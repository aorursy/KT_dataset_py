# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))
df=pd.read_csv("../input/world-university-rankings/timesData.csv")
df.head()
df.tail()
df.info()
fs=df.iloc[:100,:]

fs.head()
trace1=go.Scatter(

                  x=fs.world_rank,

                  y=fs.citations,

                  mode='lines',

                  name='citations',

                  marker=dict(color='rgba(16,112,2,0.8)'),

                  text=fs.university_name)

trace2=go.Scatter(

                  x=fs.world_rank,

                  y=fs.teaching,

                  mode='lines+markers',

                  name='teaching',

                  marker=dict(color='rgba(200,48,28,0.5)'),

                  text=fs.university_name)

data=[trace1,trace2]

layout=dict(title='citation and teaching vs world rank of top 100 universities',

            xaxis=dict(title="World Rank",ticklen=5, zeroline=False)

           )

fig=dict(data=data, layout=layout)

iplot(fig)

turk=df[df['country']=='Turkey']

turk
trace1=go.Scatter(

                  x=turk.world_rank,

                  y=turk.citations,

                  mode='lines',

                  name='citations',

                  marker=dict(color='rgba(16,112,2,0.8)'),

                  text=turk.university_name)

trace2=go.Scatter(

                  x=turk.world_rank,

                  y=turk.teaching,

                  mode='lines+markers',

                  name='teaching',

                  marker=dict(color='rgba(200,48,28,0.5)'),

                  text=turk.university_name)

data=[trace1,trace2]

layout=dict(title='Citation and Teaching vs World Rank of top Turkish Universities',

            xaxis=dict(title="World Rank",ticklen=5, zeroline=False)

           )

fig=dict(data=data, layout=layout)

iplot(fig)
ddd2014=df[df['year']==2014].iloc[:100,:]

ddd2014
df2014=df[df.year==2014].iloc[:100,:]

df2015=df[df.year==2015].iloc[:100,:]

df2016=df[df.year==2016].iloc[:100,:]



trace1=go.Scatter(

                  x=df2014.world_rank,

                  y=df2014.citations,

                  mode='markers',

                  name='2014',

                  marker=dict(color='rgba(200,10,10,0.6)'),

                  text=df2014.university_name

                  )

trace2=go.Scatter(

                  x=df2015.world_rank,

                  y=df2015.citations,

                  mode='markers',

                  name='2015',

                  marker=dict(color='rgba(10,200,10,0.6)'),

                  text=df2015.university_name

                  )

trace3=go.Scatter(

                  x=df2016.world_rank,

                  y=df2016.citations,

                  mode='markers',

                  name='2016',

                  marker=dict(color='rgba(10,10,200,0.8)'),

                  text=df2016.university_name

                  )

data=[trace1,trace2,trace3]

layout=dict(title="Citations vs World Rank of top 100 universities with 2014,2015 and 2016 years",

            xaxis=dict(title='Worls Rank', ticklen=5, zeroline=False),

            yaxis=dict(title='Citations', ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
df2014=df[df.year==2014].iloc[:100,:]

df2015=df[df.year==2015].iloc[:100,:]

df2016=df[df.year==2016].iloc[:100,:]



trace1=go.Scatter(

                  x=df2014.world_rank,

                  y=df2014.total_score,

                  mode='markers',

                  name='2014',

                  marker=dict(color='rgba(200,10,10,0.6)'),

                  text=df2014.university_name

                  )

trace2=go.Scatter(

                  x=df2015.world_rank,

                  y=df2015.total_score,

                  mode='markers',

                  name='2015',

                  marker=dict(color='rgba(10,200,10,0.6)'),

                  text=df2015.university_name

                  )

trace3=go.Scatter(

                  x=df2016.world_rank,

                  y=df2016.total_score,

                  mode='markers',

                  name='2016',

                  marker=dict(color='rgba(10,10,200,0.8)'),

                  text=df2016.university_name

                  )

data=[trace1,trace2,trace3]

layout=dict(title="Citations vs World Rank of top 100 universities with 2014,2015 and 2016 years",

            xaxis=dict(title='Worls Rank', ticklen=5, zeroline=False),

            yaxis=dict(title='Citations', ticklen=5,zeroline=False))

fig=dict(data=data,layout=layout)

iplot(fig)
df2014=df[df.year==2014].iloc[:3,:]

df2014
trace1=go.Bar(

              x=df2014.university_name,

              y=df2014.citations,

              name='citations',

              marker=dict(color='rgba(255,234,123,0.5)',line=dict(color='rgb(0,0,0)', width=1.5)),

              text=df2014.country

              )

trace2=go.Bar(

              x=df2014.university_name,

              y=df2014.teaching,

              name='teaching',

              marker=dict(color='rgba(200,100,44,0.5)', line=dict(color='rgb(0,0,0)', width=1.5)),

              text=df2014.country

              )

data=[trace1,trace2]

layout=go.Layout(barmode='group')

fig=go.Figure(data=data, layout=layout)

iplot(fig)
df2014=df[df.year==2014].iloc[:3,:]

x=df2014.university_name

trace1={

    'x':x,

    'y':df2014.citations,

    'name':'citations',

    'type':'bar'

};

trace2={

    'x':x,

    'y':df2014.teaching,

    'name':'teaching',

    'type':'bar'

};

data=[trace1,trace2];

layout={

    'xaxis':{'title':'Top 3 Universities'},

    'barmode':'relative',

    'title':'Citations and Teaching of top 3 universities in 2014'

};



fig=go.Figure(data=data, layout=layout)

iplot(fig)
df2014=df[df.year==2014].iloc[:3,:]



trace1={

    'x':df2014.university_name,

    'y':df2014.citations,

    'name':'citations',

    'type':'bar'

};

trace2={

    'x':df2014.university_name,

    'y':df2014.teaching,

    'name':'teaching',

    'type':'bar'

};

data=[trace1,trace2];

layout={

    'xaxis':{'title':'Top 3 Universities'},

    'barmode':'relative',

    'title':'Citations and Teaching of top 3 universities in 2014'

};



fig=go.Figure(data=data, layout=layout)

iplot(fig)
df2016=df[df.year==2016].iloc[:7,:]

pie=df2016.num_students

pie_list=[float(each.replace(',','.')) for each in df2016.num_students]

labels=df2016.university_name



fig={

  'data':[

        {

          'values':pie_list,

          'labels':labels,

          'domain':{'x':[0, .5]},

          'name':'Number of students rates',

          'hoverinfo': 'label+percent+name',

          'hole': .3,

          'type':'pie'      

        },],

  'layout':{

        'title':'Universities Number of Students rates',

        'annotations':[

            {'font':{'size':20},

             'showarrow':False,

             'text':'Number of Students',

             'x':0.20,

             'y':1

            },

         ]

      }                              

}    



iplot(fig)
df2016=df[df.year==2016].iloc[:7,:]

pie=df2016.num_students

pie_list=[float(each.replace(',','.')) for each in df2016.num_students]

labels=df2016.university_name



fig = {

  "data": [

    {

      "values": pie_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Universities Number of Students rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Number of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}



iplot(fig)
df2016=df[df.year==2016].iloc[:7,:]

df2016
df2016=df[df.year==2016].iloc[:20,:]

num_students_size=[float(each.replace(',','.')) for each in df2016.num_students]

international_color=[float(each) for each in df2016.international]

data=[

    {

    'y':df2016.teaching,

    'x':df2016.world_rank,

    'mode':'markers',

    'marker':{

        'color':international_color,

        'size':num_students_size,

        'showscale': True     

    },

    'text':df2016.university_name

    }

]

iplot(data)
x2011=df.student_staff_ratio[df.year==2011]

x2012=df.student_staff_ratio[df.year==2012]

trace1=go.Histogram(

      x=x2011,

      opacity=0.75,

      name='2011',

      marker=dict(color='rgba(120,39,35,0.5)'))

trace2=go.Histogram(

      x=x2012,

      opacity=0.75,

      name='2012',

      marker=dict(color='rgba(20,190,23,0.5)'))

data=[trace1,trace2]

layout=go.Layout(barmode='overlay',

                 title='students-staff ratio in 2011 and 2012', 

                 xaxis=dict(title='students-staff ratio'),

                 yaxis=dict(title='Count'),

                )

fig=go.Figure(data=data, layout=layout)

iplot(fig)

x2011=df.country[df.year==2011]

plt.subplots(figsize=(8,8))

wordcloud=WordCloud(

                    background_color='white',

                    width=512,

                    height=384

                   ).generate(' '.join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# data prepararion

x2011 = df.country[df.year == 2011]

plt.subplots(figsize=(8,8))

wordcloud=WordCloud(

                          background_color='white',

                          width=512,

                         height=384

                          ).generate(' '.join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x2015=df[df.year==2015]

x2015
x2015=df[df.year==2015]

trace0=go.Box(

              y=x2015.total_score,

              name='total score of universities in 2015',

              marker=dict(color='rgb(140,20,39)',)

             )

trace1=go.Box(

              y=x2015.research,

              name='research of universities in 2015',

              marker=dict(color='rgb(20,120,39)',)

              )

data=[trace0,trace1]

iplot(data)
dataframe=df[df.year==2015]

data2015=dataframe.loc[:,['research','international','total_score']]

data2015['index']=np.arange(1,len(data2015)+1)

fig=ff.create_scatterplotmatrix(data2015, 

                                diag='box',

                                index='index', 

                                colormap='Portland',

                                colormap_type='cat',

                                height=700, width=700

                                )

iplot(fig)
dataframe=df[df.year==2015]

data2015=dataframe.loc[:,['research','international','total_score']]

data2015['index']=np.arange(1,len(data2015)+1)

data2015['index']
trace1=go.Scatter(

     x=dataframe.world_rank,

     y=dataframe.teaching,

     name='teaching',

     marker=dict(color='rgba(160,30,20,0.8)'),

                 )

trace2=go.Scatter(

     x=dataframe.world_rank,

     y=dataframe.income,

     name='income',

     marker=dict(color='rgba(90,180,10,0.7)'),

                )

data=[trace1,trace2]

layout=go.Layout(

       xaxis2=dict(

              domain=[0.6,0.95],

              anchor='y2',

                   ),

       yaxis=dict(

             domain=[0.6,0.95],

             anchor='x2',       

                   ),

      title='Income and Teaching vs World Rank of Universities'

                )

fig=go.Figure(data=data,layout=layout)

iplot(fig)
dataframe=df[df.year==2015]

trace1=go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(

          size=10,color='rgb(255,0,0)',

               )

  

                   )

data=[trace1]

layout=go.Layout(

        margin=dict(l=0,r=0,b=0,t=0)

                 )

fig=go.Figure(data=data,layout=layout)

iplot(fig)
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

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
trace1=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    name='research'

                 )

trace2=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    xaxis='x2',

    yaxis='y2',

    name='citations'

                 )

trace3=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x3',

    yaxis='y3',

    name='income'

                 )

trace4=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    xaxis='x4',

    yaxis='y4',

    name='total_score'

                 )

data=[trace1,trace2,trace3,trace4]

layout=go.Layout(

    xaxis=dict(

             domain=[0,0.45]

               ),

    yaxis=dict(

             domain=[0,0.45]

              ),

    xaxis2=dict(

             domain=[0.55,1]

               ),

    xaxis3=dict(

             domain=[0,0.45],

             anchor='y3'

               ),

    xaxis4=dict(

             domain=[0.55,1],

             anchor='y4'

               ),

    yaxis2=dict(

            domain=[0,0.45],

            anchor='x2'  

               ),

    yaxis3=dict(

            domain=[0.55,1]

               ),

    yaxis4=dict(

            domain=[0.55,1],

            anchor='x4'    

               ),

    title='Research, citation, income and total score vs World Rank of Universities'

                 )

    

fig=go.Figure(data=data,layout=layout)

iplot(fig)

    

    

    
trace1=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    name='research'

                 )

trace2=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    xaxis='x2',

    yaxis='y2',

    name='citations'

                 )

trace3=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x3',

    yaxis='y3',

    name='income'

                 )

trace4=go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    xaxis='x4',

    yaxis='y4',

    name='total_score'

                 )

data=[trace1,trace2,trace3,trace4]

layout=go.Layout(

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

    title='Research, citation, income and total score vs World Rank of Universities'

                 )

    

fig=go.Figure(data=data,layout=layout)

iplot(fig)

    
aerial=pd.read_csv('../input/world-war-ii/operations.csv')
aerial.head()