import numpy as np

import pandas as pd



from plotly.offline import init_notebook_mode,iplot,plot



init_notebook_mode(connected=True)

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



from wordcloud import WordCloud

import matplotlib.pyplot as plt

import os

os.listdir("../input/world-university-rankings/")
PATH = "../input/world-university-rankings/"



timeData = pd.read_csv(f"{PATH}/timesData.csv")    
timeData.info()
timeData.head()
df = timeData.iloc[:100,:]



t1 = go.Scatter(

    x= df.world_rank,

    y= df.citations,

    mode="lines",

    name="citations",

    marker = dict(color="rgba(116,116,116,0.9)")

)



t2 = go.Scatter(

    x = df.world_rank,

    y = df.teaching,

    mode= "lines+markers",

    name="teaching",

    marker = dict(color="green")

)



data = [t1,t2]



#heading and title

layout = dict(title="citations and teaching vs world rank of top 100 universities",

             xaxis=dict(title='worldrank',ticklen=5,zeroline=False)

             )



fig = dict(data=data,layout = layout)

iplot(fig)
df_2014 = timeData[timeData['year'] == 2014]

df_2015 = timeData[timeData['year'] == 2015]

df_2016 = timeData[timeData['year'] == 2016]



t1 = go.Scatter(

    x = df_2014.world_rank,

    y = df_2014.citations,

    mode = "markers",

    name = "2014",

    marker = dict(color="rgba(255,128,255,0.8)")

)



t2 = go.Scatter(

    x = df_2015.world_rank,

    y = df_2015.citations,

    mode = "markers",

    name = "2015",

    marker = dict(color="rgba(255,128,0,0.8)")

)

t3 = go.Scatter(

    x = df_2016.world_rank,

    y = df_2016.citations,

    mode = "markers",

    name = "2016",

    marker = dict(color="rgba(255,0,255,0.8)")

)



data = [t1,t2,t3]

layout = dict(title='Citation vs world_rank of top 100 university in 2014,2015,2016',

              xaxis=dict(title='world rank',ticklen=5,zeroline=True),

             yaxis=dict(title="citations",ticklen=5,zeroline=False))



fig = dict(data=data,layout=layout)

iplot(fig)
df_3 = timeData.iloc[:3,:]
t1 = go.Bar(

    x = df_3.university_name,

    y = df_3.citations,

    name="citations",

    marker=dict(color='rgba(255,116,128,0.8)',

               line=dict(color='rgba(0,0,0,0)',width=1.5))

)



t2 = go.Bar(

    x = df_3.university_name,

    y=df_3.teaching,

    name="teaching",

    marker=dict(color='rgba(125,125,125,0.8)',

           line = dict(color='rgba(0,0,0,0)',width=1.5)),



)



data = [t1,t2]



layout = dict(barmode="group",title="Bar plot of ciation and teaching of top 3 university at 2011")

fig = dict(data=data,layout=layout)

iplot(fig)
t1 = go.Bar(

    x = df_3.university_name,

    y = df_3.citations,

    name="citations",

    marker=dict(color='rgba(255,116,128,0.8)',

               line=dict(color='rgba(0,0,0,0)',width=1.5))

)



t2 = go.Bar(

    x = df_3.university_name,

    y=df_3.teaching,

    name="teaching",

    marker=dict(color='rgba(125,125,125,0.8)',

           line = dict(color='rgba(0,0,0,0)',width=1.5)),



)



data = [t1,t2]



layout = dict(barmode="relative",title="Bar plot of ciation and teaching of top 3 university at 2011")

fig = dict(data=data,layout=layout)

iplot(fig)
from plotly import subplots



df_7 = timeData.iloc[:7,:]



t1 = go.Bar(

    x = df_7.research,

    y = df_7.university_name,

    name = 'research',

    marker = dict(color='rgba(255,6,100,0.8)',

                 line=dict(color='rgba(171, 50, 96, 1.0)',width=1)),

    orientation='h'

)



t2 = go.Scatter(

    x = df_7.income,

    y = df_7.university_name,

    name="income",

    mode = "lines+markers",

    marker = dict(color="rgba(255,128,0,0.8)")

)



layout = dict(title="Income and Research",

              yaxis=dict(showticklabels=True,domain=[0,0.85]),

              yaxis2=dict(showticklabels=False,showline=True,linecolor

                          ='rgba(102, 102, 102, 0.8)',linewidth=2,domain=[0,0.85]),

              xaxis=dict(zeroline=False,showline=False,showgrid=True,showticklabels=True),

              xaxis2=dict(zeroline=False,showline=False,showgrid=True,showticklabels=True,side='top'),

              legend= dict(x=0.1,y=1.038,font=dict(size=15)),

              margin= dict(l=100,r=100,t=100,b=100),

              paper_bgcolor='rgb(255,248,255)',

              plot_bgcolor = 'rgb(255,255,255)'

             )



fig = subplots.make_subplots(rows=1,cols=2,shared_xaxes=False,shared_yaxes=True)

fig.append_trace(t1,1,1)

fig.append_trace(t2,1,2)



fig['layout'].update(layout)

iplot(fig)
df_7.head()


pie = df_7.num_students

pie_list = [float(e.replace(',','.')) for e in pie]

labels = df_7.university_name

fig = {

    "data":[

        {

            "values":pie_list,

            "labels":labels,

            "name":"number of students",

            "hole":0.3,

            "type":"pie"

        }

    ],

    "layout":{

        "title":"Piechart of Number of students "

    }

}



iplot(fig)
df_10 = timeData.iloc[:10,:]
num_students = [float(e.replace(',','.')) for e in df_10.num_students]

international_color = [float(e) for e in df_10.international]

fig = {

    "data":[

        {

            "x":df_10.world_rank,

            "y":df_10.teaching,

            "name":'world rank',

            "mode":"markers",

            "marker":{

                "color":international_color,

                "size":num_students,

                "showscale":True

            },

            "text":df_10.university_name

        }

    ],

    "layout":{

        "title":"World rank vs teaching",

        

    }

}

iplot(fig)
df_2011_score = timeData.query("year == 2011").total_score

df_2012_score = timeData.query('year == 2012').total_score



t1 = go.Histogram(

    x = df_2011_score,

    opacity=0.8,

    name = "2011"

)

t2 = go.Histogram(

    x=df_2012_score,

    opacity=0.8,

    name='2012'

)



layout = go.Layout(barmode='overlay',

                  title="2011 and 2013 total score distribution",

                  xaxis=dict(title="Student ratio"),

                  yaxis=dict(title="student score"),

                  legend= dict(x=0.9,y=0.9,font=dict(size=10)))



data = [t1,t2]

fig = dict(data=data,layout=layout)

iplot(fig)

df_2011  = timeData[timeData.year == 2011]



fig = {

    "data":[

       { 

           "y":df_2011.research,

            "name":"2011 research",

            "marker":{

                "color":'rgba(128,128,255,0.8)',

            },

             "type":"box"

       },

        {

            "y":df_2011.income,

            "name":"2011 income",

            "marker":{

                "color":'rgba(122,255,255,0.8)'

            },

            "type":"box"

        }

    ],

    

    "layout":{

        "title":"Box plot of income and research"

    }

}



iplot(fig)
import plotly.figure_factory as ff

df2015 = timeData[timeData.year == 2015].loc[:,['research','international','income']]



df2015["index"] = np.arange(1,len(df2015)+1)



fig  = ff.create_scatterplotmatrix(df2015,diag='box',index='index',colormap='Portland',

                                   colormap_type='cat',height=700,width=700)



iplot(fig)
trace1 = go.Scatter3d(

    x=df_2015.world_rank,

    y=df_2015.research,

    z=df_2015.citations,

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
df_country = timeData.country[timeData.year == 2011]



plt.figure(figsize=(10,10))

wordcloud = WordCloud(background_color="black",

                      width=512,

                      height=384,

                     ).generate(" ".join(df_country))



plt.imshow(wordcloud)

plt.axis("off")

plt.show()
df_country_year = timeData.groupby(['country','year'])[["citations","teaching"]].mean().reset_index()
fig1 = px.choropleth(df_country_year,locations="country",

                    color="teaching",

                    projection='natural earth',

                    hover_name="country",

                   locationmode='country names',

                   )



layout = dict(title="Average teaching per country")

fig1['layout'].update(layout)

fig1.show()
fig2 = px.choropleth(df_country_year,

                    color="citations",

                    locations="country",

                    locationmode="country names",

                    projection="natural earth",

                    title="Average teaching in countries")

fig2.show()
fig3 = px.choropleth(

    df_country_year,

    color="citations",

    locations="country",

    projection="natural earth",

    locationmode="country names",

    scope="europe"

)

fig3.show()