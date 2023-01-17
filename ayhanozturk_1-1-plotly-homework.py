

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

timesData = pd.read_csv("../input/timesData.csv")

# Any results you write to the current directory are saved as output.
timesData.columns
timesData.head()
# prepare data frame

df = timesData.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = "lines",

                    name = "citations",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.university_name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.university_name)

# Creating trace3

trace3 = go.Scatter(

                    x = df.world_rank,

                    y = df.student_staff_ratio,

                    mode = "lines+markers",

                    name = "student_staff_ratio",

                    marker = dict(color = 'rgba(10, 100, 80, 0.8)'),

                    text= df.university_name)

data = [trace1, trace2,trace3]

layout = dict(title = 'Citation and Teaching vs World Rank of Top 100 Universities',

              xaxis= dict(title= 'World Rank',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:100,:]

df2015 = timesData[timesData.year == 2015].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]

# import graph objects as "go"

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

iplot(fig)


timesData.international.replace(['-'],0.0,inplace = True)

timesData.income.replace(['-'],0.0,inplace = True)







timesData.international=timesData.international.astype(float)

timesData.income=timesData.income.astype(float)







# prepare data frames

df2014 = timesData[timesData.international > 50].iloc[:100,:]

df2015 = timesData[timesData.income > 50].iloc[:100,:]

df2016 = timesData[timesData.year == 2016].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "international",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2014.university_name)

# creating trace2

trace2 =go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "income",

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

iplot(fig)
# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:3,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

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

iplot(fig)
# fark data üzerindeki kürsör açıklaması struct ile birleşik yapı oldu ama bundan önce tip uyuşmazlığını giderdim

# 

df2014 = timesData[timesData.year == 2014].iloc[:3,:]



df2014.international=df2014.international.astype(str)



struct = df2014.country+" " +df2014.international

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "citations",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = struct)

# create trace2 

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = "teaching",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = struct)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#bar mod yukarıda group burda relative







# prepare data frames

df2014 = timesData[timesData.year == 2014].iloc[:3,:]

# import graph objects as "go"

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

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'citations and teaching of top 3 universities in 2014'

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)

df2016=timesData[timesData.year==2016].iloc[:7,:]

pie1_list=[float(each.replace(',','.')) for each in df2016.num_students]

pie1_list

labels = df2016.university_name




# figure

fig = {

  "data": [

    {

      "values": pie1_list,

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

              "showarrow": True,

              "text": "Number of Students",

                "x": 0.1,

                "y": 1

            },

        ]

    }

}

iplot(fig)
# dört bıyutlu gösterim yani x eksini ve y ekseni datasının yanında yuvarlağın büyüklüğü ve rengide birşey ifade ediyor ile farkl

# data preparation

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

iplot(data)
# prepare data

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

iplot(fig)
x2011=timesData.country[timesData.year==2011]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(

                          background_color='black',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# data preparation

x2015 = timesData[timesData.year == 2016]



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

iplot(data)
timesData.head()
# birden fazala feature un birbiriyle karşılaştırılmasını sağlıyor

# yani  aşağıda 3x3 matrisdeki 1x3 de total score ile resarch arasındaki bağıntı doğrusal olması gibi

# import figure factory

import plotly.figure_factory as ff

# prepare data

dataframe = timesData[timesData.year == 2016]

data2016 = dataframe.loc[:,["research","student_staff_ratio", "total_score"]]

data2016["index"] = np.arange(1,len(data2016)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2016, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=900, width=900)

iplot(fig)
dataframe["color"] = ""

dataframe.color[dataframe.research>90]=0.1

dataframe.color

dataframe
# first line plot

trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name = "teaching",

    text = dataframe.country,

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    text = dataframe.country,

    xaxis='x2',

    yaxis='y2',

    name = "income",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

# second line plot

trace3 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.student_staff_ratio,

    text = dataframe.country,

    xaxis='x3',

    yaxis='y3',

    name = "student_staff_ratio",

    marker = dict(color = 'rgba(16, 12, 20, 0.8)'),

)

data = [trace1, trace2, trace3]

layout = go.Layout(

    xaxis2=dict(

        domain=[0, 0.45],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2',

    ),

    

    xaxis3=dict(

        domain=[0.55, 1],

        anchor='y3',        

    ),

    yaxis3=dict(

        domain=[0.55, 1],

        anchor='x3',

    ),

    

    title = 'Income and Teaching vs World Rank of Universities'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

# first line plot

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

        domain=[0.5, 1],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.7, 1],

        anchor='x2',

    ),

    title = 'Income and Teaching vs World Rank of Universities'



)



fig = go.Figure(data=data, layout=layout)

iplot(fig)

dataframe = timesData[timesData.year == 2016].iloc[:20,:]

num_students_size  = [float(each.replace(',', '.')) for each in dataframe.num_students]

num_students_size=num_students_size*10

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
# yerleştirme mantığı x ekseni 0 dan 1 e kadar y eksenide 0 dan 1 e kadar tanımlamalara göre plotu 

# diktörtgen kabul edip x uzunluğu ve y uzunluğunu tanımlıyoruz

# yerleştirme mantığı 1. xaxsis x koordinatta 0 dan başla 0.45 e kadar et yaxis y de 0 dan başla 0.45 devam et

# yerleştirme mantığı 2. xaxsis x koordinatta 0.45 dan başla 1 e kadar devam et yaxis y de 0 dan başla 0.45 devam et

# not dolasıyla 1. ve 2 plot bir birinin devamı ve yükseklikleri aynı 

# yani 1. plot diktortgeni 0 dan başalayan çizgi 0.45 de bitiyor 

# ve yukarı doğru 0 dan başlayı 0.55 e kadar çıkıyor bununla 1. plot diktörgeni oluşuyor aynı şekilde 

# ikinci plot diktörtgen çizgisi 0.55 den 1 e kadar 

# çizildiğini farz et sonra yukarı doğru  0 dan başlayı 0.55 e kadar çıkıyor bununla 2. plot diktörgeni oluşuyor

# aynı şekilde 3. ve 4. plot dikdörtgeni oluşuyor 



#Not iki yukarıdaki INSET PLOT dersindeki gösterimde bu mantıkladır text = dataframe.university_name, 

#burda virgül kısmını unutma



#Not paylaşılan not dışında text yapısı eklendi

trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.research,

    text = dataframe.university_name,

    name = "research"

)

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.citations,

    text = dataframe.university_name,

    xaxis='x2',

    yaxis='y2',

    name = "citations"

)

trace3 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    text = dataframe.university_name,

    xaxis='x3',

    yaxis='y3',

    name = "income"

)

trace4 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.total_score,

    text = dataframe.university_name,

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

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'Research, citation, income and total score VS World Rank of Universities'

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)