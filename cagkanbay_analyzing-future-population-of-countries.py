import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# plotly
import plotly.plotly as py
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
#seaborn
import seaborn as sns
# matplotlib
import matplotlib.pyplot as plt
#missingno
import missingno as msno
# ignore warnings
import warnings
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))
data=pd.read_csv("../input/countries of the world.csv")
data.head(10)
data.info()
data.replace(to_replace=r",",value=".",regex=True,inplace=True) # replace "," to "."
data.columns[4:] # select features which types are going to change
for each in data.columns[4:]:
    data[each]=data[each].astype("float") # change object to float
data.dtypes
msno.matrix(data)
plt.show()
data.isna().sum() # sum. of the Nan values at each feature
data[data==0].count() # sum. of the zeros at each feature
data.columns[7:] # select the features which is going to change
for each in data.columns[7:]:
    data[each].replace(to_replace=0,value="NaN",inplace=True) # replace zeros to NaN values
for each in data.columns[7:]:
    data[each]=data[each].astype("float") # again making objects to float 
data.dropna(inplace=True) #dropping NaN values
data.index = range(len(data.index)) # rearange index numbers
data.info()
trace = [go.Bar(
            x=data.Region.value_counts().index,
            y=data.Region.value_counts().values,
            text=data.Region.value_counts().values,
            hoverinfo = 'text',
            textposition = 'auto',
            marker = dict(color = 'rgba(253,174,97, 0.5)',
                             line=dict(color='rgb(0,200,200)',width=1.5)),
    )]

layout = dict(
    title = 'Number of Countries by Region',
)
fig = go.Figure(data=trace, layout=layout)
iplot(fig)
data.groupby("Region")["Population"].sum() 
fig = {
  "data": [
    {
      "values": data.groupby("Region")["Population"].sum().values,
      "labels": data.groupby("Region")["Population"].sum().index,
      "domain": {"x": [0, .5]},
      "name": "Population Rate",
      "hoverinfo":"label+percent+name",
      "hole": .3,
      "type": "pie"
    },],
  "layout": {
        "title":"Rate of Population by Regions",
        "annotations": [
            { "font": { "size": 20},
              "showarrow": False,
              "text": "Regions",
                "x": 0.20,
                "y": 1.1
            },
        ]
    }
}
iplot(fig)
trace = dict(type='choropleth',
            locations = data.Country,
            locationmode = 'country names', z = data.Population,
            text = data.Country, colorbar = {'title':'Population'},
            colorscale = 'Hot', reversescale = True)

layout = dict(title='Population of Countries',
geo = dict(showframe=False,projection={'type':'natural earth'}))
choromap = go.Figure(data = [trace],layout = layout)
iplot(choromap,validate=False)
def rate(row):
    return (row["Birthrate"]+row["Net migration"]-row["Deathrate"])/10
data["Growth Rate%"]=data.apply(rate,axis=1) # crating new column
data.groupby("Region")["Population"].max()
max_pop=data.loc[data.Population.isin(data.groupby("Region")["Population"].max().values)]
max_pop
def linear(row):
    return row["Population"]*((row["Growth Rate%"]/100)+1)
max_pop["Next Year Pop."]=max_pop.apply(linear,axis=1) 
import math # math library for calculation
def expo(row):
    return (row["Population"]*(math.exp(row["Growth Rate%"]/100)))
max_pop["Next Year Pop. Exp."]=max_pop.apply(expo,axis=1)
max_pop[["Country","Population","Next Year Pop.","Next Year Pop. Exp."]]
trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop.Population,
                name = "Current Population",
                marker = dict(color = 'rgba(0, 255, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["Next Year Pop."],
                name = "Next Year Pop.",
                marker = dict(color = 'rgba(255, 255, 0, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Next Year Population by Constant Growth")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)
trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop.Population,
                name = "Current Population",
                marker = dict(color = 'rgba(0, 255, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["Next Year Pop. Exp."],
                name = "Next Year Pop.",
                marker = dict(color = 'rgba(255, 0, 255, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Next Year Population by Continuous Growth")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)
def linear5(row):
    return row["Population"]*math.pow(((row["Growth Rate%"]/100)+1),5)
max_pop["After 5 Year Pop."]=max_pop.apply(linear5,axis=1)
import math
def expo5(row):
    return row["Population"]*math.pow((math.exp(row["Growth Rate%"]/100)),5)
max_pop["After 5 Year Pop. Exp."]=max_pop.apply(expo5,axis=1)
max_pop[["Country","Population","After 5 Year Pop.","After 5 Year Pop. Exp."]]
trace1 = go.Bar(
                x = max_pop.Country,
                y = max_pop["After 5 Year Pop."],
                name = "Constant Growth",
                marker = dict(color = 'rgba(500,100,150, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
trace2 = go.Bar(
                x = max_pop.Country,
                y = max_pop["After 5 Year Pop. Exp."],
                name = "Continuous Growth",
                marker = dict(color = 'rgba(150,100,500, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)),
                text = max_pop.Country)
data1 = [trace1, trace2]
layout = go.Layout(barmode = "group",title="Populations After 5 Year")
fig = go.Figure(data = data1, layout = layout)
iplot(fig)
f, ax = plt.subplots(figsize=(16, 16))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data.corr(), cmap=cmap, vmax=.3, center=0,square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5},fmt= '.1f')
plt.show()
data2=data.loc[:,["GDP ($ per capita)","Literacy (%)","Climate","Agriculture","Industry","Service","Growth Rate%"]]
data2["index"]=np.arange(1,len(data2)+1)
fig = ff.create_scatterplotmatrix(data2, diag='box', index='index',colormap='Portland',
                                  colormap_type='cat',title="Growth Factors",
                                  height=1200, width=1200)
iplot(fig)
def Intensity(row):
    return (row["Pop. Density (per sq. mi.)"]/row["Crops (%)"])*100
max_pop["Physiological Intensity(sq.mi.)"]=max_pop.apply(Intensity,axis=1)
max_pop[["Country","Physiological Intensity(sq.mi.)"]]
trace0 = go.Scatter(
    x=max_pop["Physiological Intensity(sq.mi.)"],
    y=max_pop["Population"],
    text=max_pop.Country ,
    mode='markers',
    marker=dict(
        colorbar = {'title':'Crops (%)'},
        color=max_pop["Crops (%)"],
        size=max_pop["Pop. Density (per sq. mi.)"],
        showscale=True
    )
)

data3 = [trace0]
layout = go.Layout(
    title='Physiological Intensity v. Population by v. Pop. Density by Crops% ',
    xaxis=dict(
        title='Physiological Intensity',
    ),
    yaxis=dict(
        title='Population',
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig = go.Figure(data = data3, layout = layout)
iplot(fig)