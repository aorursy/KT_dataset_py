# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Datamızı TimesData değişkenine aktarıyoruz.

TimesData = pd.read_csv('../input/timesData.csv')
TimesData.info()  #En basit analiz metodu.
TimesData.head() #İlk 5 girdiye bakarak data hakkında bilgi sahibi olmamız gerekli.
#Line Plot --- Citation and teaching vs World rank of top 100 universities



df = TimesData.iloc[:100,:]  #İlk 100 Universiteyi alırız öncelikle



# import graph objects as "go"

import plotly.graph_objs as go



#Trace1 oluşturuyoruz.

trace1 = go.Scatter(

                    x = df.world_rank,

                    y = df.citations,

                    mode = "lines",    #Tabloda çizgi şeklinde görülsün.

                    name = "citations",   #Labeldeki yazacak metin. 

                    marker = dict(color='rgba(16,112,2,0.8)'),

                    text= df.university_name )    #Fare ile gezinirken gözükecek metin.



#Trace2 oluşturuyoruz.

trace2 = go.Scatter(

                    x = df.world_rank,

                    y = df.teaching,

                    mode = "lines+markers",

                    name = "teaching",

                    marker = dict(color='rgba(80,26,80,0.8)'),

                    text = df.university_name)



#Trace1 ve trace2 yi tek bir datada birleştiriyoruz.

data = [trace1,trace2]



layout = dict(title = "Citation and teaching vs World rank of top 100 universities",

              xaxis = dict(title='World Ranking',ticklen = 5,zeroline = False))



fig = dict(data = data, layout = layout)

iplot(fig)
#Scatter Plot --- Citation vs world rank of top 100 university with 2014,2015 and 2016.

#İstenilen yıllardaki üniversite bilgilerini ayrı ayrı dataframelere ayırdık.

df2014 = TimesData[TimesData.year == 2014].iloc[:100,:]

df2015 = TimesData[TimesData.year == 2015].iloc[:100,:]

df2016 = TimesData[TimesData.year == 2016].iloc[:100,:]
#Ayrı yıllar için ayrı ayrı traceler oluşturuyorum.

trace1 = go.Scatter(

                    x = df2014.world_rank,

                    y = df2014.citations,

                    mode = "markers",

                    name = "2014",

                    marker = dict(color = 'rgba(255,128,255,0.8)'),

                    text = df2014.university_name

                    )

trace2 = go.Scatter(

                    x = df2015.world_rank,

                    y = df2015.citations,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255,128,2,0.8)'),

                    text = df2015.university_name

                    )

trace3 = go.Scatter(

                    x = df2016.world_rank,

                    y = df2016.citations,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(0,128,200,0.8)'),

                    text = df2016.university_name

                    )



data = [trace1,trace2,trace3]



layout = dict(title = "Citation vs world rank of top 100 university with 2014,2015 and 2016",

              xaxis = dict(title ="World Rank",ticklen=5,zeroline=False),

              yaxis = dict(title ="Citation",ticklen=5,zeroline=False)

             )



fig = dict(data = data,layout = layout)

iplot(fig)
#Bar Chart --- Citation and teaching of top 3 universities in 2014

df2014 = TimesData[TimesData.year == 2014].iloc[:3,:]



#Önce üniversitelerin Citationları alınır.

trace1 = go.Bar(

                x = df2014.university_name,

                y = df2014.citations,

                name = "Citations",

                marker = dict(color = 'rgba(255,128,255,0.8)',

                              line = dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country

                )



#Sonra üniversitelerin Teachingleri alınır.

trace2 = go.Bar(

                x = df2014.university_name,

                y = df2014.teaching,

                name = "Teaching",

                marker = dict(color = 'rgba(255,255,128,0.5)',

                              line = dict(color='rgb(0,0,0)',width=1.5)),

                text = df2014.country

                )



#Yukarda aldığımız bilgiler datada toplanır. Layout ile düzenlemeler yapılır.

data = [trace1,trace2]

layout = go.Layout(barmode="group")



#Tamamlanmış tüm bilgiler fig değişkeninde toplanır ve çizdirilir.

fig = go.Figure(data = data, layout = layout)

iplot(fig)
#Pie Chart -- Student rate of top 7 universities in 2016.



#2016'nın ilk 7 üniversitesinin yer aldığı bir df oluşturduk.

df2016 = TimesData[TimesData.year == 2016].iloc[:7,:]



#Öğrenci numarasındaki sayılar virgülle ayrılmış ve object olarak tanımlanmış.

#Önce numaraları arasındaki virgülleri noktaya çeviriyoruz ve float türünde pie1_liste yazdırıyoruz.

pie1 = df2016.num_students

pie1_list = [float(each.replace(',','.')) for each in pie1]



labels = df2016.university_name



#Görselleştirme

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

              "showarrow": False,

              "text": "Number of Students",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
#University world rank (first 20) vs teaching score with number of students(size) and international score (color) in 2016

df2016 = TimesData[TimesData.year == 2016].iloc[:20,:]

num_student_size = [float(each.replace(',','.')) for each in df2016.num_students]

international_color = [float(each) for each in df2016.international]





data = [

    {

        'y' : df2016.teaching,

        'x' : df2016.world_rank,

        'mode' : 'markers',  #Nokta nokta göstermesi için markers diyoruz.

        'marker' : {

            'color' : international_color,   #İnternational oranına göre renk ayarla

            'size' : num_student_size,   #Boyutunu öğrenci sayısına göre ayarla

            'showscale' : True

        },

        "text" : df2016.university_name   #Fareyi üzerine getirince gösterilecek isim.

    }

]

iplot(data)
#Histogram Plot --- Look at histogram of students-staff ratio in 2011 and 2012 years.

df2011 = TimesData.student_staff_ratio[TimesData.year == 2011]

df2012 = TimesData.student_staff_ratio[TimesData.year == 2012]



Trace1 = go.Histogram(

    x=df2011,

    opacity=0.75,

    name="2011",

    marker=dict(color='rgba(12,50,196,0.6)'))



Trace2 = go.Histogram(

    x=df2012,

    opacity=0.75,

    name="2012",

    marker=dict(color='rgba(171,50,96,0.6)'))



data = [Trace1,Trace2]

layout = go.Layout(barmode='overlay',

                  title='students-staff ratio in 2011 and 2012 years',

                  xaxis=dict(title='student-staff ratio'),

                  yaxis=dict(title='Count'),

)

fig = go.Figure(data=data,layout=layout)



iplot(fig)
#Word Cloud --- 2011 de en çok hangi ülkeden bahsedilmiş.

x2011 = TimesData.country[TimesData.year == 2011]



#Figure boyutunu ayarlamak için kullanılır.

plt.subplots(figsize=(8,8))



#Wordcloud kütüphanesi kullanımı bu şekildedir.

wordcloud = WordCloud(

    background_color="white",

    width=512,

    height=384,

).generate(" ".join(x2011))  #Kelimeleri birbirinden ayırmak için kullanılır.



plt.imshow(wordcloud) 

plt.axis('off')

plt.show()
#Box Plot --- 

x2015 = TimesData[TimesData.year == 2015]



trace0 = go.Box(

    y=x2015.total_score,

    name='total score of universities in 2015',

    marker=dict(

        color='rgb(12,12,140)',

    )

)

trace1 = go.Box(

    y=x2015.research,

    name='research of universities in 2015',

    marker=dict(

        color='rgb(12,128,128)',

    )

)

data = [trace0,trace1]

iplot(data)
#Scatter Plot Matrix -- 2 Kolon arasındaki kovaryans yani ilişkiyi görmek için kullanılır.

import plotly.figure_factory as ff



dataframe = TimesData[TimesData.year == 2015]

data2015 = dataframe.loc[:,["research","international","total_score"]]

data2015["index"] = np.arange(1,len(data2015)+1)



fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index', colormap='Portland',

                                  colormap_type='cat',height=700,width=700)

iplot(fig)
#İnset Plot --- 2 Tane plot iç içe -- Üniversitelerin teaching ve income skorlarını karşılaştır.



#Büyük olan tablonun içine koyulması gerekenler.

trace1 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.teaching,

    name="teaching",

    marker=dict(color='rgba(16,112,2,0.8)')

)



#Küçük olan tablonun içine koyulması gerekenler.

trace2 = go.Scatter(

    x=dataframe.world_rank,

    y=dataframe.income,

    xaxis='x2',

    yaxis='y2',

    name='income',

    marker=dict(color='rgba(160,112,20,0.8)')

)



#Standart birleştirme işlemleri vs.

data=[trace1,trace2]



layout=go.Layout(xaxis2=dict(domain=[0.6,0.95],anchor='y2'),

                 yaxis2=dict(domain=[0.6,0.95],anchor='x2'),

                 title='Income and teaching vs World rank of Universities')



fig=go.Figure(data=data,layout=layout)

iplot(fig)
#3D Scatter plot with Colorscaling ---



trace1 = go.Scatter3d(

    x=dataframe.world_rank,

    y=dataframe.research,

    z=dataframe.citations,

    mode='markers',

    marker=dict(

        size=10,

        color='rgba(128,50,243,0.7)'

    )

)



data=[trace1]

layout=go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0,

    )

)

fig = go.Figure(data=data,layout=layout)

iplot(fig)
#Multiple Subplots ---



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

iplot(fig)