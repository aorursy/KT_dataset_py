!pip install chart-studio
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#plotly libs

import chart_studio.plotly as py

from plotly.offline import init_notebook_mode , iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)





#wordcloud libs

from wordcloud import WordCloud



#matplotlib



import matplotlib.pyplot as plt







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
timesData = pd.read_csv("/kaggle/input/world-university-rankings/timesData.csv")
timesData.info()
timesData.head(10)
# Creating Trace 1



df = timesData.iloc[:100,:]





trace1 = go.Scatter( x = df.world_rank,

           y= df.citations,

           mode = "lines",

           name = "citations",

           marker = dict(color = 'rgba(16,112,2,0.8)'),

           text = df.university_name)

# Creating Trace 2



trace2 = go.Scatter( x = df.world_rank,

           y= df.teaching,

           mode = "lines+markers",

           name = "teaching",

           marker = dict(color = 'rgba(80,26,80,0.8)'),

           text = df.university_name)

data = [trace1,trace2]



layout  = dict(title = 'Citation and Teaching vs World Rank  of Top 100 Universities',

              xaxis = dict(title = 'World Rank',ticklen = 5, zeroline = False)

              )



fig = dict(data = data,layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]





trace1 = go.Scatter(x = df2014.world_rank,

                    y = df2014.citations,

                    mode = 'markers',

                    name = '2014',

                    marker = dict(color = 'rgba(255,128,255,1)'),

                    text = df2014.university_name

                )



trace2 = go.Scatter(x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = '2015',

                    marker = dict(color = 'rgba(255,128,2,1)'),

                    text = df2015.university_name

                   

                   )

trace3 = go.Scatter(x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = '2016',

                    marker = dict(color = 'rgba(0,255,200,1)'),

                    text = df2015.university_name

                   

                   )



data = [trace1,trace2,trace3]

layout = dict(title = "Citation vs World Rank of Top 100 Universities in 2014,2015,2016 years",

             xaxis = dict(title = "World Rank",ticklen = 5,zeroline = False),

             yaxis = dict(title = "Citations",ticklen = 5,zeroline = False) 

            

             )

            

fig = dict(data = data,layout = layout)

iplot(fig)
df2014 = timesData[timesData.year == 2014].iloc[:3,:]



trace1 = go.Bar(



                x = df2014.university_name,

                y = df2014.citations,

                name = 'citations',

                marker = dict(color = 'rgba(0,174,255,0.5)',line = dict(color = 'rgb(0,255,0)',width = 1.5)),

                text = df2014.country)



trace2 = go.Bar(



                x = df2014.university_name,

                y = df2014.teaching,

                name = 'teaching',

                marker = dict(color = 'rgba(150,0,100,0.5)',line = dict(color = 'rgb(255,0,0)',width = 1.5)),

                text = df2014.country)



data = [trace1,trace2]



layout = go.Layout(barmode = "group",yaxis = dict(title = 'Citations',ticklen=5),title = "CITATIONS AND TEACHING VALUES ACCORDING TO TOP 3 UNIVERSITIES IN THE WORLD")

fig = go.Figure(data = data,layout=layout)

iplot(fig)
# Different way to plot bar graph.



x = df2014.university_name



trace1 = {

    

    'x':x,

    'y':df2014.citations,

    'name':'citation',

    'type':'bar',

    'text':df2014.country

}



trace2 = {

    'x' : x,

    'y' : df2014.teaching,

    'name' : 'teaching',

    'type' : 'bar',

    'text':df2014.country

    

}

data = [trace1,trace2]

layout = {

    

    'xaxis' : {'title': 'Top 3 Universities'},

    'barmode' : 'relative',

    'title' : 'CITATIONS AND TEACHING VALUES ACCORDING TO TOP 3 UNIVERSITIES IN THE WORLD'

}



fig = go.Figure(data = data,layout = layout)

iplot(fig)
# Data preparation

df2016 = timesData[timesData.year == 2016].iloc[:7,:]

pie1 = df2016.num_students

pie1_list = [float(each.replace(',','')) for each in pie1]

labels = df2016.university_name



#Figure





trace = {

    

    'values' : pie1_list,

    'labels' : labels,

    'domain' : {'x' : [1,1]},

    'name' : 'Number Of Student Rates', # to put in 'hoverinfo' feature.

    'hoverinfo' : 'label+percent',# info to be shown when the cursor hovers on a piece of pie

    'hole' : .1, # size of center hole.

    'type' : 'pie'

    

    

}



data = [trace]



layout = {

    

            'title' : 'Universities Number of Students Rates',

    

    

            'annotations' : [

                

                        {'font' : {'size' : 20},

                         

                         'showarrow' : False,

                         'text' : 'Number of Students',

                         'x' : 0.50, # coordinates of graph text

                         'y' : -0.1

                        

                        },

                        

                

                

                

                

            ]

    

    

    

}

fig = dict(data = data , layout = layout)

iplot(fig)

          
df2016 = timesData[timesData.year == 2016].iloc[:20,:]

norm_value = max([float(i.replace(',','')) for i in df2016.num_students.values])/100 # scaling process

df2016


num_students_size = [float(each.replace(',',''))/norm_value for each in df2016.num_students]

international_color = [float(each) for each in df2016.international]



data = [ {

    

        'y' : df2016.teaching,

        'x' : df2016.world_rank,

        'mode' : 'markers',

        'marker' : {'color' : international_color, 'size' : num_students_size, 'showscale' : True},

        'text' : df2016.university_name

}]



iplot(data)



x2011 = timesData.student_staff_ratio[timesData.year == 2011]

x2012 = timesData.student_staff_ratio[timesData.year == 2012]



trace_1 = go.Histogram(



            x = x2011,

            opacity = 0.75,

            name = '2011',

            marker = dict(color = 'rgba(171,50,96,0.6)'))



trace_2 = go.Histogram(



            x = x2012,

            opacity = 0.75,

            name = '2012',

            marker = dict(color = 'rgba(12,50,196,0.6)'))

data = [trace_1,trace_2]



layout = go.Layout(barmode = 'overlay',title='Students-Staff ratio in 2011 and 2012 years', 

                  xaxis = dict(title = 'Students - Staff Ratio'),

                  yaxis = dict(title = 'Count')

                  )

fig = go.Figure(data = data,layout = layout)



iplot(fig)



x2011 = timesData.country[timesData.year == 2011]

plt.figure(figsize=(20,20))

wordcloud = WordCloud(

            

            background_color = 'black',

            width = 1024,

            height = 900)

wordcloud.generate(" ".join(x2011))



plt.imshow(wordcloud)

plt.axis('off')

plt.show()
x2015 = timesData[timesData.year == 2015]



trace0 = go.Box(



            y = x2015.total_score,

            name = 'Total Score of Universities in 2015',

            marker = dict(color = 'rgb(12,12,140)'))



trace1 = go.Box(



            y = x2015.research,

            name = 'Research of Universities in 2015',

            marker = dict(color = 'rgb(12,128,128)'))



data = [trace0,trace1]

iplot(data)
import plotly.figure_factory as ff



dataframe = timesData[timesData.year == 2015]

data2015 = dataframe.loc[:,["research","international","total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)



fig = ff.create_scatterplotmatrix(data2015,diag='box',index='index',colormap='Portland',colormap_type = 'cat',height=1000,width=1000)



iplot(fig)

# Firs line plot

trace1 = go.Scatter(

                    

    x = dataframe.world_rank,

    y = dataframe.teaching,

    name='teaching',

    text = dataframe.university_name,

    marker = dict(color = 'rgba(16,112,2,0.8)'))



# Second line plot

trace2 = go.Scatter(

            

    x = dataframe.world_rank,

    y = dataframe.income,

    xaxis = 'x2',

    yaxis = 'y2',

    name = 'income',

    text = dataframe.university_name,

    marker = dict(color = 'rgba(160,112,20,0.8)'))



data = [trace1,trace2]

layout = go.Layout(



        xaxis2 = dict(

            

            domain = [0.6,0.95],

            anchor = 'y2'),

            

    

        yaxis2 = dict(

                

            domain = [0.6,0.95],

            anchor = 'x2'),

    

        xaxis = dict(title = "World Rank"),

        yaxis = dict(title = "Teaching"),

    

        title = 'Income and Teaching vs World Rank of Universities')









fig = go.Figure(data = data,layout = layout)

iplot(fig)
scale_list1 = []

scale_list2 = []

for i in dataframe.world_rank.values:

          

    i = i.split('-')

    

    

    if len(i)>1:

        i[1] = float(i[1])

        scale_list2.append(i[1])

        

  

    i[0] = float(i[0])

    scale_list1.append(i[0])

    

scale_list1.extend(scale_list2)

scale_list1
trace1 = go.Scatter3d(

        

        x = dataframe.world_rank,

        y = dataframe.research,

        z = dataframe.citations,

        mode='markers',

        marker = dict(size = 10,color = scale_list1,colorscale = 'aggrnyl')

)

data = [trace1]

layout = go.Layout(margin = dict(l=0,r=0,b=0,t=0))



fig = go.Figure(data = data,layout = layout)

iplot(fig)
trace1 = go.Scatter(



            x = dataframe.world_rank,

            y = dataframe.research,

            name = 'research',

)



trace2 = go.Scatter(

            

            x = dataframe.world_rank,

            y = dataframe.citations,

            xaxis = 'x2',

            yaxis = 'y2',

            name = 'citations',

)



trace3 = go.Scatter(



            x = dataframe.world_rank,

            y = dataframe.income,

            xaxis = 'x3',

            yaxis = 'y3',

            name = 'income'

)



trace4 = go.Scatter(

        

        x = dataframe.world_rank,

        y = dataframe.total_score,

        xaxis = 'x4',

        yaxis = 'y4',

        name = 'total score'

)



data = [trace1,trace2,trace3,trace4]

layout = go.Layout(

                

            xaxis = dict(domain = [0,0.45]),

            yaxis = dict(domain = [0,0.45]),

            

            xaxis2 = dict(domain = [0.55,1]),

            xaxis3 = dict(domain = [0,0.45],anchor='y3'),

            xaxis4 = dict(domain = [0.55,1],anchor='y4'),

            

            yaxis2 = dict(domain = [0,0.45],anchor='x2'),

            yaxis3 = dict(domain = [0.55,1]),

            yaxis4 = dict(domain = [0.55,1],anchor='x3'),

    

            title = 'Research,citation,income and total score vs world rank of universities'

)



fig = go.Figure(data = data,layout = layout)

iplot(fig)
aerial = pd.read_csv("/kaggle/input/world-war-ii/operations.csv")
aerial[aerial.Country.isnull()].dropna(inplace=True) 
aerial["color"] = ""

aerial.color[aerial.Country == "USA"] = "rgb(0,116,217)"

aerial.color[aerial.Country == "GREAT BRITAIN"] = "rgb(255,65,54)"

aerial.color[aerial.Country == "NEW ZEALAND"] = "rgb(133,20,75)"

aerial.color[aerial.Country == "SOUTH AFRICA"] = "rgb(255,133,27)"



data = [dict(

    type='scattergeo',

    lon = aerial['Takeoff Longitude'],

    lat = aerial['Takeoff Latitude'],

    hoverinfo = 'text',

    text = "Country: " + aerial.Country + " Takeoff Location: "+aerial["Takeoff Location"]+" Takeoff Base: " + aerial['Takeoff Base'],

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = "antiquewhite",

        opacity = 0.7),

)]

layout = dict(

    title = 'Countries Take Off Bases ',

    hovermode='closest',

    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, projection=dict(type='mercator'),

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
data = pd.read_csv("../input/earthquake-database/database.csv")

data.head()
data.drop([3378,7512,20650],inplace=True)

data["year"] = [int(each.split("/")[2]) for each in data.iloc[:,0]]
data.head()
dataset = data.loc[:,["Date","Latitude","Longitude","Type","Depth","Magnitude","year"]]

dataset.head()
years = [str(each) for each in list(data.year.unique())]  # str unique years

# make list of types

types = ['Earthquake', 'Nuclear Explosion', 'Explosion', 'Rock Burst']

custom_colors = {

    'Earthquake': 'rgb(189, 2, 21)',

    'Nuclear Explosion': 'rgb(52, 7, 250)',

    'Explosion': 'rgb(99, 110, 250)',

    'Rock Burst': 'rgb(0, 0, 0)'

}

# make figure

figure = {

    'data': [],

    'layout': {},

    'frames': []

}



figure['layout']['geo'] = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,

               countrywidth=1, 

              landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

figure['layout']['hovermode'] = 'closest'

figure['layout']['sliders'] = {

    'args': [

        'transition', {

            'duration': 400,

            'easing': 'cubic-in-out'

        }

    ],

    'initialValue': '1965',

    'plotlycommand': 'animate',

    'values': years,

    'visible': True

}

figure['layout']['updatemenus'] = [

    {

        'buttons': [

            {

                'args': [None, {'frame': {'duration': 500, 'redraw': False},

                         'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                'label': 'Play',

                'method': 'animate'

            },

            {

                'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                'transition': {'duration': 0}}],

                'label': 'Pause',

                'method': 'animate'

            }

        ],

        'direction': 'left',

        'pad': {'r': 10, 't': 87},

        'showactive': False,

        'type': 'buttons',

        'x': 0.1,

        'xanchor': 'right',

        'y': 0,

        'yanchor': 'top'

    }

]



sliders_dict = {

    'active': 0,

    'yanchor': 'top',

    'xanchor': 'left',

    'currentvalue': {

        'font': {'size': 20},

        'prefix': 'Year:',

        'visible': True,

        'xanchor': 'right'

    },

    'transition': {'duration': 300, 'easing': 'cubic-in-out'},

    'pad': {'b': 10, 't': 50},

    'len': 0.9,

    'x': 0.1,

    'y': 0,

    'steps': []

}



# make data

year = 1695

for ty in types:

    dataset_by_year = dataset[dataset['year'] == year]

    dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]

    

    data_dict = dict(

    type='scattergeo',

    lon = dataset['Longitude'],

    lat = dataset['Latitude'],

    hoverinfo = 'text',

    text = ty,

    mode = 'markers',

    marker=dict(

        sizemode = 'area',

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        color = custom_colors[ty],

        opacity = 0.7),

)

    figure['data'].append(data_dict)

    

# make frames

for year in years:

    frame = {'data': [], 'name': str(year)}

    for ty in types:

        dataset_by_year = dataset[dataset['year'] == int(year)]

        dataset_by_year_and_cont = dataset_by_year[dataset_by_year['Type'] == ty]



        data_dict = dict(

                type='scattergeo',

                lon = dataset_by_year_and_cont['Longitude'],

                lat = dataset_by_year_and_cont['Latitude'],

                hoverinfo = 'text',

                text = ty,

                mode = 'markers',

                marker=dict(

                    sizemode = 'area',

                    sizeref = 1,

                    size= 10 ,

                    line = dict(width=1,color = "white"),

                    color = custom_colors[ty],

                    opacity = 0.7),

                name = ty

            )

        frame['data'].append(data_dict)



    figure['frames'].append(frame)

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': year,

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)





figure["layout"]["autosize"]= True

figure["layout"]["title"] = "Earthquake"       



figure['layout']['sliders'] = [sliders_dict]



iplot(figure)