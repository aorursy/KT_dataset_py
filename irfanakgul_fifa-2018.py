# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt 

from collections import Counter

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from wordcloud import WordCloud



# plotly

# import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# import WordCloud





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("/kaggle/input/fifa-18-demo-player-dataset/CompleteDataset.csv")

data['Age'] = data['Age'].astype(int)
type(data)
data.info()
# gereksiz kolonlari silme



data.drop(['Flag', 'Photo', "Club Logo","Special"], axis=1, inplace=True)
#datanin ilk 10 verisi

data.head(10)

data.columns
# prepare data frame

data1 = data.iloc[:,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = data1.Age, 

                    y = data1.Potential, # 

                    mode = "lines",  #cizgi turu

                    name = "Potential",# cubugun sonunda yazacak isim

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'), 

                    text= data1.Name) #uzerine geldiginde ne gorunsun.

# Creating trace2

trace2 = go.Scatter(

                    x = data1.Age,

                    y = data1.Overall,

                    mode = "lines+markers", # noktalarla birlestirme yapar. ve cizgi kullanir

                    name = "Overall", # cubugun sonunda yazacak isim

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= data1.Name)

data2 = [trace1, trace2]

layout = dict(title = 'Potential and Overall at first 100 players',

              xaxis= dict(title= 'best',ticklen= 10,zeroline= False) #ticlen kalinlik, zeroline '0' dan baslamasi

             )

fig = dict(data = data2, layout = layout)

iplot(fig)

#prepare data frames

#yillara gore ilk 100 veriyi aldik

age_20 = data1[data1.Age == 20].iloc[:100,:]

age_25 = data1[data1.Age == 2015].iloc[:100,:]

age_30 = data1[data1.Age == 2016].iloc[:100,:]

# import graph objects as "go"

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = age_20.Overall,

                    y = age_20.Potential,

                    mode = "markers",

                    name = "age_20",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= age_20.Name) #uzerine geldiginde ne gorunsun.

# creating trace2

trace2 =go.Scatter(

                    x = age_25.Overall,

                    y = age_25.Potential,

                    mode = "markers",

                    name = "2015",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= age_25.Name) #uzerine geldiginde ne gorunsun.

# creating trace3

trace3 =go.Scatter(

    #x, y, mode, name, marker, text mecbur on tanimli yazilmak zorunda. 

                    x = age_30.Overall,

                    y = age_30.Potential,

                    mode = "markers",

                    name = "2016",

                    marker = dict(color = 'rgba(0, 255, 200, 0.8)'),

                    text= age_30.Name) #uzerine geldiginde ne gorunsun.

data3 = [trace1, trace2, trace3]  # olusturdugumuz veriler listeye atadik



# konumlandirmayi yapar ve isimlendirir.(layout)

layout = dict(title = '20-25-30 yaslarindaki listedeki ilk 100 oyuncunun potansiyel ve allover gorsellestirmesi',

              xaxis= dict(title= 'Overall',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Potential',ticklen= 5,zeroline= False)

             )

fig = dict(data = data3, layout = layout)

iplot(fig)
# prepare data frames

# ST si 85 ve uzeri olan tum futbolcularun yas ve hizlanma grafigidir

St_top = data1[data1.ST >= 85].iloc[:,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = St_top.Name,

                y = St_top.Acceleration,

                name = "Acceleration",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)), 

                text = St_top.Name)

# create trace2 

trace2 = go.Bar(

                x = St_top.Name,

                y = St_top.Age,

                name = "Age",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = St_top.Name)

data5 = [trace1, trace2]

layout = go.Layout(title = 'ST si 85 ve uzeri olan tum futbolcularun yas ve hizlanma grafigidir',barmode = "group") 

fig = go.Figure(data = data5, layout = layout)

iplot(fig)
# prepare data frames

age_20_less = data1[data1.Age <= 20].iloc[:20,:]

# import graph objects as "go"

import plotly.graph_objs as go



x = age_20_less.Name



trace1 = {

  'x': x,

  'y': age_20_less.Potential,

  'name': 'Potential',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': age_20_less.Overall,

  'name': 'Overall',

  'type': 'bar'

};

data6 = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 universities'},

  'barmode': 'relative',

  'title': 'yasi 20 den kucuk olan oyuncularin potansiyel ve overall grafigi(ilk 20 liste)'

};

fig = go.Figure(data = data6, layout = layout)

iplot(fig)
# data preparation

age_36_more = data1[data1.Age >= 36].iloc[:10,:]

pie1 = age_36_more.Overall

#num_Student deki 

# str(2,4) => str(2.4) = > float(2.4) = 2.4

labels = age_36_more.Name

# figure

fig = {

  "data": [

    {

      "values": pie1,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": " ",

      "hoverinfo":"label+percent+name",

      "hole": .3, #pastaanin ortasinda delik acar

      "type": "pie"

    },],

  "layout": {

        "title":"yasi 36+ overall grafigi",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Overall",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
# data preparation

# yasi 20 den kucuk oyuncul

age_20_less1 = data1[data1.Age <= 20].iloc[:20,:]



data6 = [

    {

        'y': age_20_less1.LB,

        'x': age_20_less1.Acceleration,

        'mode': 'markers',

        'marker': {

            'color': age_20_less1.Overall,

            'size': age_20_less1.Potential,

            'showscale': True

        },

        "text" :  age_20_less1.Name    

    }

]

iplot(data6)



# prepare data

#histogram cizimi

age_20 = data1.Overall[data1.Age <= 20]

age_35 = data1.Overall[data1.Age >= 35]





trace1 = go.Histogram(

    x=age_20,

    opacity=0.75,

    name = "Yasi 20-",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=age_35,

    opacity=0.75,

    name = "Yasi 35+",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data7 = [trace1, trace2]

layout = go.Layout(barmode='overlay', # ustuste koyacak ovarlay

                   title=' yasi 20- ve 35+ olanlarin Overall Grafigi',

                   xaxis=dict(title='Overall'),

                   yaxis=dict( title='sayi'),

)

fig = go.Figure(data=data7, layout=layout)

iplot(fig)
# data preparation

age_35 = data1[data1.Age >= 35]



trace0 = go.Box(

    y=age_35.ST,

    name = 'yasi 35+, ST',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=age_35.Overall,

    name = 'Yas 35+ Overall',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data8 = [trace0, trace1]

iplot(data8)
# import figure factory

import plotly.figure_factory as ff

# prepare data

df = data1[data1.Age >= 40]

all_ = df.loc[:,["ST","RB", "LB"]] #uc sutununun tamamini aldik

all_["index"] = np.arange(1,len(all_)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(all_, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
data1.corr()

f,ax = plt.subplots(figsize = (18,18)) # resmin buyuklugunu ayarlar

sns.heatmap(data1.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
data1.Overall.plot(kind = 'line',color = 'g', label = 'Overall', linewidth =1, alpha = 0.5, grid = True, linestyle = '-.')

data1.Potential.plot(color = 'r', label ='Potential', linewidth = 1, alpha = 0.5, grid = True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()