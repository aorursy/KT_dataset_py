# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

from collections import Counter 

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore') 

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
case=pd.read_csv("../input/coronavirusdataset/Case.csv")

info=pd.read_csv("../input/coronavirusdataset/PatientInfo.csv")

route=pd.read_csv("../input/coronavirusdataset/PatientRoute.csv")

time=pd.read_csv("../input/coronavirusdataset/Time.csv")

gender=pd.read_csv("../input/coronavirusdataset/TimeGender.csv")

province=pd.read_csv("../input/coronavirusdataset/TimeProvince.csv")

t_age=pd.read_csv("../input/coronavirusdataset/TimeAge.csv")

region=pd.read_csv("../input/coronavirusdataset/Region.csv")
info.info()
info.head()
info.country.unique()
info.country.value_counts()

a=info.country.value_counts().values

b=info.country.value_counts().index

plt.figure(figsize=(7,7))

sns.barplot(x=b,y=a)

plt.xlabel("Number of patients.")

plt.ylabel("Country")

plt.title("Which country has the most patitent?")
info.sex.value_counts()
male=info[info.sex=="male"]

female=info[info.sex=="female"]

import plotly.graph_objs as go

trace1=go.Bar(x=female.sex,

             y=female.sex.value_counts().values,

               name = "female",

                marker = dict(color = 'rgba(255, 174, 255, 0.8)'),

                             

                text = female.country,

             )

trace2=go.Bar(x=male.sex,

             y=male.sex.value_counts().values,

               name = "male",

                marker = dict(color = 'rgba(10, 129,90, 0.8)'),

                            

                text = male.country,

             )



data = [trace1,trace2]

layout={

  'xaxis': {'title': 'Sex'},

  'barmode': 'group',

  'title': 'Distribution of patients by gender.',

   "font": {"size":20} ,

    "width":500,

    "height":500

};

fig = go.Figure(data = data, layout = layout)

iplot(fig)



info.province.unique()
info.province.value_counts()
data = [dict(

    type="scattergeo",

    lon = region["longitude"],

    lat = region["latitude"],

    hoverinfo = "text",

    text ="Province:"+region.province,

    mode = "markers",

    marker=dict(

        sizemode = "area",

        sizeref = 1,

        size= 10 ,

        line = dict(width=1,color = "white"),

        

        opacity = 0.7),

)]

layout = dict(

    title = ' Cities With Disease',

    hovermode='closest',

    geo = dict(showframe=False, showland=True, showcoastlines=True, showcountries=True,landcolor = 'rgb(217, 217, 217)',

              subunitwidth=1,

              showlakes = True,

              lakecolor = 'rgb(255, 255, 255)',

              countrycolor="rgb(5, 5, 5)")

)







fig = go.Figure(data=data,layout=layout)

iplot(fig)
p=info.province.value_counts().index



y=info.province.value_counts().values

plt.figure(figsize=(15,10))

sns.barplot(x=p,y=y,palette="rocket")

plt.xlabel("Province")

plt.xticks(rotation=90)

plt.title("Which cities are the most diseased people in Korea?")

plt.ylabel("Infected People")



info.age.value_counts()
a=info.age.value_counts().values

b=info.age.value_counts().index

import plotly.express as px



fig=px.bar(x=b,y=a,title="What is the age range of people who are infected?")

fig.show()
death=info[info.state=="deceased"]

released=info[info.state=="released"]

isolated=info[info.state=="isolated"]

data=pd.concat([death,released,isolated])

data1=[each for each in data.state]

labels=data.state

fig = {

  "data": [

    {

      "values":labels.value_counts().values,

      "labels": labels.value_counts().index,

      "domain": {"x": [0, .8]},

      "name": "Death or Released",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],"layout": {

        "title":"What are the death rates?",

    

        "annotations": [

            { "font": { "size": 25},

             

              "showarrow": False,

                 "text":"Death and releases rates",

                "x": .4,

                "y": 1.1

            },

        ]

    }

}

iplot(fig)
death.info()

death.age.value_counts()
death["num_age"]=[2020-each for each in death.birth_year]
plt.figure(figsize=(10,10))

sns.barplot(x=death.num_age.value_counts().index,y=death.num_age.value_counts().values)

plt.ylabel("How many people died ?")

plt.xlabel("Ages of people who died")

plt.title("What are the ages of people who died ?",fontsize=20,color="red")

plt.xticks(rotation=90)
data={"y":death.num_age.value_counts().values,

      "x":death.num_age.value_counts().index,

      "mode":"markers",

      "marker":{"color":"rgb(10,10,10)"},

      "text":death.sex};

layout={"xaxis":{"title":"Age"},"title":"What are the ages of people who died ?"}

fig=go.Figure(data=data,layout=layout)

iplot(fig)
time.info()
time["month"]=[int(each.split("-")[1])for each in time.date]

January=time[time.month==1]

February=time[time.month==2]

March=time[time.month==3]

x=["January","February","March"]

a=0

for each in January.test:

    a=each+a

b=0

for each in February.test:

    b=each+b

c=0

for each in March.test:

    c=each+c    

y=[a,b,c]

df=[[x,y]]

trace=go.Scatter( x = x,

                    y = y,

                    mode = "lines+markers",

                    

                    marker = dict(color = 'rgba(240, 1, 17, 0.8)')

                    )

layout=dict(title="South Korea Monthly Tests Numbers",xaxis= dict(title= 'Month',ticklen= 5,zeroline= False),

            yaxis=dict(title="number of tests performed"))

data=[trace]

fig=dict(data=data,layout=layout)

iplot(fig)
p=0

for each in January.released:

    p=each+p

r=0

for each in February.released:

    r=each+r

t=0

for each in March.released:

    t=each+t 

d=0

for each in January.deceased:

    d=each+d

e=0

for each in February.deceased:

    e=each+e

f=0

for each in March.deceased:

    f=each+f

y3=[p,r,t]    

y2=[d,e,f]  
trace=go.Scatter( x = x,

                    y = y3,

                    mode = "lines+markers",

                    name = "released",

                    marker = dict(color = 'rgba(240, 1, 17, 0.8)')

                    )

trace2=go.Scatter(x = x,

                    y = y2,

                    mode = "lines+markers",

                    name = "deceased",

                    marker = dict(color = 'rgba(1, 1, 1, 0.8)'))

layout=dict(title="What is the number of people who die and recover by months ?",

            xaxis= dict(title= 'Month',ticklen= 5,zeroline= False))

data=[trace,trace2]

fig=dict(data=data,layout=layout)

iplot(fig)
sns.lmplot(data=time,x="deceased",y="confirmed")

plt.show()