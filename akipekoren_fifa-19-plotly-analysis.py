# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

from wordcloud import WordCloud



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/data.csv")

df.head(15)
#typical Scatter graph

new_index=(df["Overall"].sort_values(ascending=False)).index.values

sortedData=df.reindex(new_index)

best_players=sortedData.head(100)



trace1=go.Scatter(

                    x=best_players.Name,

                    y=best_players.ShortPassing,

                    mode="lines",

                    name="ShortPassing",

                    marker=dict(color="rgba(25,125,25,0.5)"),

                    text=best_players.Nationality)

trace2=go.Scatter(

                    x=best_players.Name,

                    y=best_players.Crossing,

                    mode="lines",

                    name="Crossing",

                    marker=dict(color="rgba(109,99,109,0.5)"),

                    text=best_players.Nationality)

data=[trace1,trace2]

layout=dict(title="Comparision between Crossing and ShortPassing" ,xaxis=dict(title="Best Players",ticklen=5,zeroline=False) )



fig=dict(data=data,layout=layout)



iplot(fig)



best_players.info()
#Show which goal keeper is best



best_GKs=df[df.Position=="GK"]

best_GKs["Wage"] = best_GKs["Wage"].str.replace("€","")

best_GKs["Wage"]=best_GKs["Wage"].str.replace("K","")

best_GKs["Wage"]=best_GKs["Wage"].astype(int)

best_GKs["Wage"]=best_GKs["Wage"]*1000   # With these statements we convert player wages to integer.

new_index=best_GKs["Wage"].sort_values(ascending=False).index.values

highestPaidGk=best_GKs.reindex(new_index)   #sorting  goalkeepers by their wages.



top20_best_GKs=highestPaidGk.head(20)

data=[

    {

        "y": top20_best_GKs.Overall,

        "x": top20_best_GKs.Name,

        

        "mode":"markers",

        "marker":{

            "color":highestPaidGk.GKPositioning,

            "size" :highestPaidGk.GKReflexes,

            'showscale': True

        },

    "text":top20_best_GKs.Club   

    }

]

iplot(data)

## Comparision of three club of Spain





dfBarcelona=df[df.Club=="FC Barcelona"].iloc[:20,:]

dfAtletico=df[df.Club=="Atlético Madrid"].iloc[:20,:]

dfRealMadrid=df[df.Club=="Real Madrid"].iloc[:20,:]



trace1=go.Scatter(

                x=dfBarcelona.Overall,

                y=dfBarcelona.Potential,

                mode="markers",

                name="Barca",

                marker=dict(color="rgba(255, 128, 255, 0.8)"),

                text=dfBarcelona.Name)



trace2=go.Scatter(

                x=dfAtletico.Overall,

                y=dfAtletico.Potential,

                mode="markers",

                name="Atletico",

                marker=dict(color="rgba(255, 128, 2, 0.8)"),

                text=dfAtletico.Name)



trace3=go.Scatter(

                x=dfRealMadrid.Overall,

                y=dfRealMadrid.Potential,

                mode="markers",

                name="Real",

                marker=dict(color="rgba(0, 255, 200, 0.8)"),

                text=dfRealMadrid.Name)



data=[trace1,trace2,trace3]

layout=dict(title="Atletico vs Barca vs Real", xaxis=dict(title="Players",ticklen= 5,zeroline= False),

           yaxis=dict(title="Overalls",ticklen= 5,zeroline= False))

fig=dict(data=data,layout=layout)

iplot(fig)
# With Bar plot, we can see overall and potential values on graph.



bestBarcaData=dfBarcelona[dfBarcelona.Position!="GK"].iloc[:5,:] # We can see the potential of best 5 barca players.



trace1=go.Bar(

            x=bestBarcaData.Name,

            y=bestBarcaData.Overall,

            name="Overall",

            marker=dict(color="rgba(120,15,15,0.5)"),

            text=bestBarcaData.Nationality)

trace2=go.Bar(

            x=bestBarcaData.Name,

            y=bestBarcaData.Potential,

            name="Potential",

            marker=dict(color="rgba(55,55,55,0.5)"),

            text=bestBarcaData.Nationality)

data=[trace1,trace2]

layout=dict(barmode="group")

fig=go.Figure(data=data, layout=layout)

iplot(fig)



# We can show which theam has more player on the top 100 list(best_players), larger words mean more.



x2011=best_players.Club

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(x2011))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
# Also we can work with 3D graphics



best_players["Unnamed: 0"].astype(int)

best_players["ranking"]=best_players["Unnamed: 0"]+1



trace1=go.Scatter3d(

                    x=best_players["ranking"],

                    y=best_players["Penalties"],

                    z=best_players["FKAccuracy"],

                    mode="markers",

                    marker=dict(color="rgba(125,125,5,0.5)"))



data=[trace1]



layout=go.Layout(

                margin=dict(l=0,

                            r=0,

                            t=0,

                            b=0))





fig=dict(data=data,layout=layout)



iplot(fig)





### With using Box Plot we can make a comparision between Barcelona and Real Madrid, We can easily their max,min and median values.



trace1=go.Box(

            y=dfBarcelona.Overall,

            name="Barcelona Datas",

            marker=dict(color="rgb(12,5,155,0.5)")

            )

trace2=go.Box(

            y=dfRealMadrid.Overall,

            name="Real Madrid Datas",

            marker=dict(color="rgb(155,200,5,0.5)"))



data=[trace1,trace2]

iplot(data)