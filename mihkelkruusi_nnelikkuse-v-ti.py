import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



%matplotlib inline

pd.set_option('display.max_rows', 20)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_17 = pd.read_csv("../input/2017.csv")
df_17.head(20)
data = dict(type = "choropleth", 

           locations = df_17["Country"],

           locationmode = "country names",

           z = df_17["Happiness.Rank"], 

           text = df_17["Country"],

           colorscale = "Greys", reversescale = True)

layout = dict(title = "Õnnelikkus maailmas 2017", 

             geo = dict(showframe = False, 

                       projection = {"type": "Mercator"}))

choromap = go.Figure(data = [data], layout=layout)

iplot(choromap)
df_15 = pd.read_csv("../input/2015.csv")

df_16 = pd.read_csv("../input/2016.csv")

df_17=df_17.rename(columns = {'Happiness.Score':'Happiness_Score'})

df_16=df_16.rename(columns = {'Happiness Score':'Happiness_Score'})

df_15=df_15.rename(columns = {'Happiness Score':'Happiness_Score'})

df_15["Year"] = "2015"

df_16["Year"] = "2016"

df_17["Year"] = "2017"

df4 = pd.concat([df_15[["Country","Happiness_Score","Year"]],df_16[["Country","Happiness_Score","Year"]],df_17[["Country","Happiness_Score","Year"]]])

df4.groupby("Year")["Happiness_Score"].mean().plot(title="Keskmine õnnelikkuse skoor (aastatel 2015, 2016 ja 2017)");
df4.groupby("Year").aggregate({"Happiness_Score": ["mean", "median","min", "max"]})
df_17["Happiness_Score"].plot.hist(title="Õnnelikkuse skoori jaotus (2017)", grid=True, rwidth=0.95);
df_17.plot.scatter("Economy..GDP.per.Capita.","Happiness_Score", marker="$ € $", alpha=0.3,s=200,color="purple", title="Õnnelikkuse sõltuvus riigi majanduslikust olukorrast");