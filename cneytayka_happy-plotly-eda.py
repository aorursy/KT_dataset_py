# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
h2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
h2015.head()
#Datamızda işlem yapabilmek için columnların isimleri üzerinde degişiklik yaptık.



h2015["Happiness_Rank"]=h2015["Happiness Rank"].values

h2015.drop(["Happiness Rank"],axis=1,inplace=True)



h2015["Economy"]=h2015["Economy (GDP per Capita)"].values

h2015.drop(["Economy (GDP per Capita)"],axis=1,inplace=True)



h2015["Happiness_Score"]=h2015["Happiness Score"].values

h2015.drop(["Happiness Score"],axis=1,inplace=True)



h2015["Healt"]=h2015["Health (Life Expectancy)"].values

h2015.drop(["Health (Life Expectancy)"],axis=1,inplace=True)



h2015["Trust"]=h2015["Trust (Government Corruption)"].values

h2015.drop(["Trust (Government Corruption)"],axis=1,inplace=True)



h2015["Dystopia_Residual"]=h2015["Dystopia Residual"].values

h2015.drop(["Dystopia Residual"],axis=1,inplace=True)



h2015["Standard_Error"]=h2015["Standard Error"].values

h2015.drop(["Standard Error"],axis=1,inplace=True)
h2015.head()
# example: Economy and Family vs Happiness Rank 



import plotly.graph_objs as go





trace1=go.Scatter(

                    x=h2015.Happiness_Rank,

                    y=h2015.Economy,

                    name="Economy",

                    mode="lines",

                    marker=dict(color="rgba(15,50,100,0.8)"),

                    text=h2015.Region

)



trace2=go.Scatter(

                    x=h2015.Happiness_Rank,

                    y=h2015.Family,

                    name="Family",

                    mode="lines+markers",

                    marker=dict(color="rgba(180,20,10,0.5)"),

                    text=h2015.Region

)



data=[trace1,trace2]



layout=dict(title="Economy and Family vs Happiness Rank",

           xaxis= dict(title= 'Happiness Rank',ticklen= 7,zeroline= False)

           )



fig = dict(data = data, layout = layout)

iplot(fig)
#freedom and Healt vs Happiness_rank







import plotly.graph_objs as go



# create trace1 



trace1=go.Scatter(

                    x=h2015.Happiness_Rank,

                    y=h2015.Freedom,

                    name="Freedom",

                    mode="markers",

                    marker=dict(color="rgba(255,50,80,0.8)"),

                    text=h2015.Region

)



trace2=go.Scatter(

                    x=h2015.Happiness_Rank,

                    y=h2015.Healt,

                    name="Healt",

                    mode="markers",

                    marker=dict(color="rgba(10,95,210,0.5)"),

                    text=h2015.Region

)



data=[trace1,trace2]



layout=dict(title="Freedom and Healt vs Happiness Rank",

           xaxis= dict(title= 'Happiness Rank',ticklen= 7,zeroline= False),

            yaxis=dict(title="Count",ticklen=6,zeroline=True)

           )



fig = dict(data = data, layout = layout)

iplot(fig)
h2015.Region.value_counts()

# family and Freedom vs Western Europe top 3 



x1=h2015[h2015.Region=="Western Europe"].iloc[:20,:]

import plotly.graph_objs as go

trace1 = go.Bar(

                x =x1.Country ,

                y =x1.Family,

                name = "Family",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                text = x1.Happiness_Score)

# create trace2 

trace2 = go.Bar(

                x = x1.Country,

                y = x1.Freedom,

                name = "Freedom",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=1.5)),

                text = x1.Happiness_Score)

data = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data, layout = layout)

iplot(fig)
h2015.head()
b1=h2015[h2015.Region=="Sub-Saharan Africa"].iloc[:,:]

print(b1)
# figure

b1=h2015[h2015.Region=="Sub-Saharan Africa"].iloc[:,:]

values=b1.Happiness_Score

labels=b1.Country

fig = {

  "data": [

    {

      "values": values,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "country of Happiness Score",

      "hoverinfo":"label+percent",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Country Regio of Happiness Ratio",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Country",

                "x": 0.20,

                "y": 0.55

            },

        ]

    }

}

iplot(fig)
h2015.head()
# create trace 1 that is 3d scatter

trace1 = go.Scatter3d(

    x=h2015.Happiness_Rank,

    y=h2015.Region,

    z=h2015.Family,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(90,160,250)',                # set color to an array/list of desired values      

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
h2015.head()
h2015[h2015.Region=="North America"]
import plotly.figure_factory as ff

# prepare data

dataframe = h2015[h2015.Region=="North America"]

data2015 = dataframe.loc[:,["Family","Economy", "Healt"]]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=750, width=750)

iplot(fig)
x2 = h2015.Country.head(35)

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()