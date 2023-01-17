# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



#import plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# matplotlib

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from collections import Counter

import os

print(os.listdir("../input"))

import sys



# Any results you write to the current directory are saved as output.
# import data

df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv")
df.head(5)
# information about data

df.info()
for x in df.columns:

    sys.stdout.write(str(x)+", ")  
df.rename(columns={"Serial No.":"Serial_No","GRE Score":"GRE","TOEFL Score":"TOEFL",

                   "University Rating":"UnivRaiting","LOR ":"LOR","Chance of Admit ":"Chance_Admit"},inplace=True)
df.head()
# This is not correct data for visualisation with plotly. It's just for learning.

# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.UnivRaiting,

                    y = df.SOP,

                    mode = "lines",

                    name = "SOP",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text = df.UnivRaiting)



# Creating trace2

trace2 = go.Scatter(

                    x = df.UnivRaiting,

                    y = df.LOR,

                    mode = "lines+markers",

                    name = "LOR",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text = df.UnivRaiting)



data = [trace1,trace2]

layout = dict(title = "SOP and LOR vs Universities' Raiting",

             xaxis=dict(title = "University' Raiting",ticklen = 5,zeroline=False)

             )

fig = dict(data=data,layout=layout)

iplot(fig)
counter_univ = Counter(df.UnivRaiting)

counter_univ
# close warnings

import warnings

warnings.filterwarnings("ignore")





# This is not correct data for visualisation with plotly. It's just for learning.

#prepare data frames

df_1 = df[df.UnivRaiting==1]

df_2 = df[df.UnivRaiting==2].iloc[:34,:]

df_3 = df[df.UnivRaiting==3].iloc[:34,:]

df_4 = df[df.UnivRaiting==4].iloc[:34,:]

df_5 = df[df.UnivRaiting==5].iloc[:34,:]



# For correctly comparing we should add new columns which is new serial number

new_serial=[]

for i in range(34):

    new_serial.append(i)

df_1["new_serial"]=new_serial

df_2["new_serial"]=new_serial

df_3["new_serial"]=new_serial

df_4["new_serial"]=new_serial

df_5["new_serial"]=new_serial







# creating trace1

trace1 = go.Scatter(

                    x = df_1.new_serial,

                    y = df_1.Chance_Admit,

                    mode = "markers",

                    name = "Univs number 1",

                    marker = dict(color = 'rgba(255, 128, 255, 1)'),

                    text = df_1.Chance_Admit)

trace2 = go.Scatter(

                    x = df_2.new_serial,

                    y = df_2.Chance_Admit,

                    mode = "markers",

                    name = "Univs number 2",

                    marker = dict(color = 'rgba(255, 128, 2, 1)'),

                    text = df_2.Chance_Admit)

trace3 = go.Scatter(

                    x = df_3.new_serial,

                    y = df_3.Chance_Admit,

                    mode = "markers",

                    name = "Univs number 3",

                    marker = dict(color = 'rgba(0, 255, 200, 1)'),

                    text = df_3.Chance_Admit)

trace4 = go.Scatter(

                    x = df_4.new_serial,

                    y = df_4.Chance_Admit,

                    mode = "markers",

                    name = "Univs number 4",

                    marker = dict(color = 'rgba(100, 200, 110, 1)'),

                    text = df_4.Chance_Admit)

trace5 = go.Scatter(

                    x = df_5.new_serial,

                    y = df_5.Chance_Admit,

                    mode = "markers",

                    name = "Univs number 5",

                    marker = dict(color = 'rgba(150, 55, 255, 0.8)'),

                    text = df_5.Chance_Admit)

data = [trace1,trace2,trace3,trace4,trace5]



layout = dict(title = "Chance Admit vs Serial Number of 34 Universities with Raitings",

              xaxis = dict(title = 'Serial Number',ticklen = 5,zeroline = False),

              yaxis = dict(title = 'Chance Admit',ticklen = 5,zeroline = False)

             )

fig = dict(data=data,layout=layout)

iplot(fig)
df.head()
# prepare dataframes

df_1 = df.iloc[:5,:]

# creating trace1

trace1 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.GRE,

                name = "GRE",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                              line = dict(color = 'rgb(0,0,0)',width=1.5)),

                text = df_1.UnivRaiting)

# Creating trace2 

trace2 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.TOEFL,

                name = "TOEFL",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line = dict(color = "rgb(0,0,0)",width=1.5)),

                text = df_1.UnivRaiting)

data = [trace1,trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data,layout=layout)

iplot(fig)
# prepare data frames

df_1 = df.iloc[:5,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.LOR,

                name = "LOR",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_1.UnivRaiting)

# create trace2 

trace2 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.SOP,

                name = "SOP",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = df_1.UnivRaiting)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
# prepare data frames

df_1 = df.iloc[:5,:]



# creating trace1

trace1 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.LOR,

                name = "LOR")

trace2 = go.Bar(

                x = df_1.Serial_No,

                y = df_1.SOP,

                name = "SOP")



data = [trace1,trace2]



layout = dict(title = "LOR and SOP of First Five Universities",

              xaxis = dict(title = "Firs 5 Universities"),

              barmode = "relative")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
counter_univ1 = Counter(df.UnivRaiting)

counter_univ
# First we create new empty dataframe

df_count = pd.DataFrame()



# we create new dictionary

counter_univ1 = dict(counter_univ)



univ_no = []

counts = []

# Now we can insert data to our dataframe

for i,j in counter_univ1.items():

    univ_no.append(i)

    counts.append(j)

df_count["univ_no"]=univ_no

df_count["counts"] = counts



labels = df_count.univ_no

pie_list = counts



#figura

fig = {

    "data": [

        {

            "values": pie_list,

            "labels": labels,

            "domain": {"x": [0, .5]},

            "name": "Number of Universities Raiting",

            "hoverinfo": "label+percent+name",

            "hole": .3,

            "type": "pie"

        }],

    "layout": {

        "title": "Universities Number of Rates",

        "annotations": [

            {

                "font": {"size": 20},

                "showarrow": True,

                "text": "Number of Universities",

                "x": 0.20,

                "y": 1

            }

        ]

    }

    

}



iplot(fig)
# data preparetion

df_20 = df.iloc[:20,:]

num_univ_size = [each for each in df_20.CGPA]

univ_color = [each for each in df_20.Chance_Admit]



data = [

    {

        "y": df_20.GRE,

        "x": df_20.Serial_No,

        "mode": "markers",

        "marker": {

            "color": univ_color,

            "size": num_univ_size,

            "showscale":True

        },

        "text":df_20.UnivRaiting

    }

]

iplot(data)
# preparing data

x2 = df.CGPA[df.UnivRaiting==2]

x4 = df.CGPA[df.UnivRaiting==4]



trace1 = go.Histogram(

        x = x2,

        opacity = 0.8,

        name="Univ Number 2",

        marker = dict(color = "rgba(171, 50, 96, 0.6)"))

trace2 = go.Histogram(

        x = x4,

        opacity = 0.8,

        name="Univ Number 4",

        marker = dict(color = "rgba(12, 50, 196, 0.6)"))



data = [trace1,trace2]



layout = go.Layout(barmode="overlay",

                  title = "2 Number Universities vs 4 Number",

                  xaxis = dict(title = "University Number"),

                  yaxis = dict(title = "count"))

fig = go.Figure(data = data,layout=layout)



iplot(fig)
# data preparation

x2 = df[df.UnivRaiting==2]



trace1 = go.Box(

    y = x2.SOP,

    name = "SOP of University Number 2",

    marker = dict(

                    color = "rgb(12,12,140)"))

trace2 = go.Box(

    y = x2.LOR,

    name = "LOR of University Number 2",

    marker = dict(

                    color = "rgb(12, 128, 128)"))

data = [trace1,trace2]

iplot(data)
df.head()
# import figure factory

import plotly.figure_factory as ff



# preparing data

df_matrix = df.loc[:,["GRE","TOEFL","CGPA","Chance_Admit"]]

df_matrix["index"] = np.arange(1,len(df_matrix)+1)



# scatter matrix

fig = ff.create_scatterplotmatrix(df_matrix,diag="box",index="index",

                                      colormap="Portland",colormap_type="cat",

                                      height=850,width=850)

iplot(fig)
# create trace1 that is 3d scatter

trace1 = go.Scatter3d(

                        x = df.GRE,

                        y = df.TOEFL,

                        z = df.CGPA,

                        mode = "markers",

                        marker = dict(

                                        size = 10,

                                        color = "rgb(255,0,0)"))

data = [trace1]



layout = go.Layout(

                    margin = dict(

                                    l = 0,

                                    r = 0,

                                    b = 0,

                                    t = 0))



fig = go.Figure(data = data,layout=layout)

iplot(fig)