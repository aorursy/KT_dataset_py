import numpy as np

#NumPy is a python library used for working with arrays.

#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.



import pandas as pd 

#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 

#a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?

#Does column A correlate with column B?

#What does the distribution of data in column C look like?

#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria

#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.

#Store the cleaned, transformed data back into a CSV, other file or database



import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.



import matplotlib.pyplot as plt

#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

plt.style.use("seaborn-whitegrid")

#plt.style.available : To see all the available style in matplotlib library



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")

timesData.info()
timesData.head()
#Line Charts Example: Citation and Teaching vs World Rank of Top 100 Universities



# prepare data frame

df = timesData.iloc[:100, :]



import plotly

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.world_rank,   #x axis

                    y = df.citations,    #y axis

                    mode = "lines",      #type of plot like marker, line or line + markers

                    name = "citations",  #name of the plots

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.university_name)   #The hover text



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



fig = dict(data = data, layout = layout)   # fig = it includes data and layout

plotly.offline.iplot(fig)                               #iplot() = plots the figure(fig) that is created by data and layout
import plotly.express as px 

import numpy as np  



# function of numpy.random  

np.random.seed(42)  

    

random_x= np.random.randint(1,101,100)  

random_y= np.random.randint(1,101,100)  

    

plot = px.scatter(random_x, random_y) 

plot.show()
#Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years



# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]



import plotly

import plotly.graph_objs as go



# creating trace1

trace1 =go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2014.university_name)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2015.university_name)

# creating trace3

trace3 =go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= df2016.university_name)



data = [trace1, trace2, trace3]



layout = dict(title = 'Citation vs world rank of top 100 universities with 2014, 2015 and 2016 years',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Citation',ticklen= 5,zeroline= False)

             )



fig = dict(data = data, layout = layout)

plotly.offline.iplot(fig) 
import plotly.express as px 

import numpy  as np

  

# creating random data through randomint  

np.random.seed(42)  

    

random_x= np.random.randint(1,101,100)  

random_y= np.random.randint(1,101,100) 

  

fig = px.bar(random_x, random_y) 

fig.show()
# Citations and teaching of top 3 universities in 2014 

# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:3,:]

df2014
import plotly

import plotly.graph_objs as go



df2014 = timesData[timesData.year == 2014].iloc[:3,:]



# create trace1 

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1)),

                text = df2014.country)



# create trace2 

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = "teaching",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country)



data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

plotly.offline.iplot(fig) 
df2014 = timesData[timesData.year == 2014].iloc[:3,:]

import plotly

import plotly.graph_objs as go



x = df2014.university_name



trace1 = {

  'x': x,

  'y': df2014.citations,

  'name': 'citation',

  'type': 'bar'

};



trace2 = {

  'x': x,

  'y': df2014.teaching,

  'name': 'teaching',

  'type': 'bar'

};



data = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 Universities'},

  'barmode': 'relative',

  'title': 'Citations and Teaching of top 3 Universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

plotly.offline.iplot(fig) 
# CITATIONS AND INCOME



import plotly

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt



df2016 = timesData[timesData.year == 2016].iloc[:7,:]



y_saving     =  [each for each in df2016.research]

y_net_worth  =  [float(each) for each in df2016.income]

x_saving     =  [each for each in df2016.university_name]

x_net_worth  =  [each for each in df2016.university_name]



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

                title='Citations and income',

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

plotly.offline.iplot(fig) 
import plotly.express as px 

import numpy as np

  

# Random Data 

random_x = [100, 2000, 550] 

names = ['A', 'B', 'C'] 

  

fig = px.pie(values=random_x, names=names) 

fig.show()
df2016.info()

# OBJECT > FLOAT
df2016.head()

#Num_students    , >>>> .   (float number 19.919 etc)
# Students rate of top 7 universities in 2016



import plotly



df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students



pie1_list = [float(each.replace(',', '.')) for each in df2016.num_students]  # str(2,4) => str(2.4) = > float(2.4) = 2.4

labels = df2016.university_name





fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Number Of Students Rates",

      "hoverinfo":"label+percent+name",

      "hole": .05,

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

plotly.offline.iplot(fig) 
import plotly.graph_objects as px 

import numpy as np 

np.random.seed(42) 

random_x= np.random.randint(1,101,100)  

random_y= np.random.randint(1,101,100) 

plot = px.Figure(data=[px.Scatter( 

    x = random_x, 

    y = random_y, 

    mode = 'markers', 

    marker_size = [115, 20, 30]) 

])                  

plot.show()
# University world rank (first 20) vs teaching score with number of students(size) and international score (color) in 2016



df2016 = timesData[timesData.year == 2016].iloc[:20,:]

num_students_size  = [float(each.replace(',', '.')) for each in df2016.num_students]

international_color = [float(each) for each in df2016.international]



data = [

    {

        'y': df2016.teaching,

        'x': df2016.world_rank,

        'mode': 'markers',

        'marker': {

            'color': international_color,

            'size': num_students_size,

            'showscale': True

        },

        "text" :  df2016.university_name    

    }

]



plotly.offline.iplot(data) 
# Students-staff ratio in 2011 and 2012 

x2011 = timesData.student_staff_ratio[timesData.year == 2011]

x2012 = timesData.student_staff_ratio[timesData.year == 2012]



trace1 = go.Histogram(

    x=x2011,

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=x2012,

    opacity=0.75,

    name = "2012",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' students-staff ratio in 2011 and 2012',

                   xaxis=dict(title='students-staff ratio'),

                   yaxis=dict( title='Count'),

)



fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig) 
import matplotlib.pyplot as plt

from wordcloud import WordCloud 



x2011 = timesData.country[timesData.year == 2011]

plt.subplots(figsize=(10,10)) 



wordcloud = WordCloud(

                          background_color='black',

                          width=600,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
import plotly

import plotly.graph_objs as go

from plotly import tools

import matplotlib.pyplot as plt



x2015 = timesData[timesData.year == 2015]



trace0 = go.Box(

    y=x2015.total_score,

    name = 'total score of universities in 2015',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=x2015.research,

    name = 'research of universities in 2015',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace0, trace1]

plotly.offline.iplot(data) 
#it helps us to see covariance and relation between more than 2 features

import plotly.figure_factory as ff



dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international", "total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)



# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

plotly.offline.iplot(fig) 
# 2 plots are in one frame



trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name = "teaching",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),  

)



# second line plot 

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',

    yaxis='y2',

    name = "income",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)



data = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

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

plotly.offline.iplot(fig) 
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

plotly.offline.iplot(fig) 
import plotly.graph_objects as go 

import numpy as np 

  

x1 = np.linspace(-4, 4, 9)  

y1 = np.linspace(-5, 5, 11)  

z1 = np.linspace(-5, 5, 11)  

  

X, Y, Z = np.meshgrid(x1, y1, z1) 

  

values = (np.sin(X**2 + Y**2))/(X**2 + Y**2) 

  

fig = go.Figure(data=go.Volume( 

    x=X.flatten(), 

    y=Y.flatten(), 

    z=Z.flatten(), 

    value=values.flatten(), 

    opacity=0.1, 

    )) 

  

fig.show()
import plotly.express as px 

fig = px.treemap( 

    names = ["A","B", "C", "D", "E"], 

    parents = ["Plotly", "A", "B", "C", "A"] 

) 

  

fig.show()
# Multiple Subplots: While comparing more than one features, multiple subplots can be useful.



trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    name = "research"

)

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    xaxis='x2',

    yaxis='y2',

    name = "citations"

)

trace3 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x3',

    yaxis='y3',

    name = "income"

)

trace4 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    xaxis='x4',

    yaxis='y4',

    name = "total_score"

)



data = [trace1, trace2, trace3, trace4]

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

fig = go.Figure(data=data, layout=layout)

plotly.offline.iplot(fig) 
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import plotly.express as px 

  

gapminder = px.data.gapminder() 

gapminder.head()

#we have obtained data from almost all countries in the world from 1952 to 2007 

#with fields like life expectancy, GDP per capita and population.  
import plotly.express as px 



gapminder = px.data.gapminder() 

gapminder.head(30) 

  

fig = px.choropleth(gapminder, 

                    locations ="iso_alpha", 

                    color ="lifeExp", 

                    hover_name ="country",  

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="world", 

                    animation_frame ="year") 

fig.show()
#scope refers to the area of scope of the choropleth. 

#For example, if we type scope=”asia”, the following is displayed:

import plotly.express as px 

gapminder = px.data.gapminder() 

gapminder.head(30) 

fig = px.choropleth(gapminder, 

                    locations ="iso_alpha", 

                    color ="lifeExp", 

                    hover_name ="country",  

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="asia", 

                    animation_frame ="year") 

fig.show()
import plotly.express as px  

gapminder = px.data.gapminder() 

gapminder.head(15) 

fig = px.bar(gapminder,  

             x ="continent",  

             y ="pop", 

             color ='lifeExp', 

             animation_frame ='year', 

             hover_name ='country',  

             range_y =[0, 4000000000]) 

fig.show()
import plotly.express as px 

gapminder = px.data.gapminder() 

gapminder.head(15) 

  

fig = px.density_contour(gapminder,  

                         x ="gdpPercap",  

                         y ="lifeExp",  

                         color ="continent",  

                         marginal_y ="histogram", 

                         animation_frame ='year',  

                         animation_group ='country',  

                         range_y =[25, 100]) 

fig.show()
import plotly.express as px   

gapminder = px.data.gapminder() 

gapminder.head(15) 

fig = px.scatter( 

    gapminder,  

    x ="gdpPercap",  

    y ="lifeExp",  

    animation_frame ="year",  

    animation_group ="country", 

    size ="pop",  

    color ="continent",  

    hover_name ="country",  

    facet_col ="continent", 

    size_max = 45, 

    range_y =[25, 90] 

) 

fig.show()