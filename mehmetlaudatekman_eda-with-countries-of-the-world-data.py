# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns #Seaborn visualization library

import matplotlib.pyplot as plt #Matplotlib visualization library







"""

Plotly Visualization Library / Main Tool of This Kernel

"""

import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px









# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")
data.head()
data.tail()
data.columns
data.dtypes
data.info()
#Matrix plot

import missingno as missin #I am importing the library that I use

data = pd.read_csv("/kaggle/input/countries-of-the-world/countries of the world.csv")

fig,ax = plt.subplots(figsize=(8,8)) #I am creating my figure

missin.matrix(data,ax=ax,sparkline=False) # I am creating the plot

plt.show() 
#Missingno bar plot

fig,ax = plt.subplots(figsize=(8,8))

missin.bar(data,ax=ax)

plt.show()
data.dropna(inplace=True)
def comma2dot(dataframe,feature):

    

    dirty_data = [i for i in dataframe[feature]]

    clean_data = []

    for i in dirty_data:

        

        if "," in i:

            i = i.replace(",",".")

        

        clean_data.append(i)

        

    dataframe[feature] = clean_data    

    return dataframe
comma_including_ftrs = ["Net migration","Infant mortality (per 1000 births)","Literacy (%)"

                          ,"Phones (per 1000)","Arable (%)","Crops (%)","Coastline (coast/area ratio)"

                          ,"Birthrate","Deathrate"]



for cm in comma_including_ftrs :

    

    data = comma2dot(data,cm)
def type_converter(dataframe,feature,dtype="float"):

    

    if dtype == "float":

        dataframe[feature] = dataframe[feature].astype(float)

    

    elif dtype == "int":

        dataframe[feature] = dataframe[feature].astype(int)

    

    elif dtype == "string" or dtype == "str":

        dataframe[feature] = dataframe[feature].astype(str)

    

    return dataframe
data.columns

features = ["Population","Area (sq. mi.)","Net migration","Infant mortality (per 1000 births)","Literacy (%)",

           "Phones (per 1000)","Arable (%)","Crops (%)","Coastline (coast/area ratio)","Birthrate","Deathrate"]



for ftr in features:

        

    data = type_converter(data,ftr)

    
data.corr()
fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),linewidth=1,annot=True,linecolor="Black",fmt=".1f")

plt.title("Correlation Map",fontsize=15)

plt.show()
#We are going to use this threshold for filtering correlation values

threshold = 0.6 



#Dataframe stacking

corr = data.corr()

stacked = corr.stack().reset_index()

stacked.columns = ["feature1","feature2","cor"]



#Filtering correlation values

stacked = stacked.loc[(stacked.feature1 != stacked.feature2) & (stacked.cor >= threshold)]

stacked = stacked.reset_index()

stacked

import networkx as nx

network_graph = nx.from_pandas_edgelist(stacked,"feature1","feature2")

nx.draw_circular(network_graph,with_labels=True,node_color="blue",node_size=20,edge_color="#C95DD9",linewidths=13)

#Histogram

trace1 = go.Histogram(x=data["Population"]

                     ,name="Population"

                     ,marker=(dict(color="rgba(234,12,43,0.8)"))

                     ,opacity=0.7)



layout = go.Layout(title="Population Histogram"

                  ,xaxis=dict(title="Population")

                  ,yaxis=dict(title="Counts"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
normalized_df = data.copy()

normalized_df["Population"] = normalized_df["Population"]/max(normalized_df["Population"])

normalized_df["Area (sq. mi.)"] = normalized_df["Area (sq. mi.)"]/max(normalized_df["Area (sq. mi.)"])



new_indexes=normalized_df["Population"].sort_values(ascending=False).index.values

normalized_df = normalized_df.reindex(new_indexes)



trace1 = go.Scatter(x=normalized_df["Population"],

                    y=normalized_df["Area (sq. mi.)"],

                    mode="markers",

                    name = "Population",

                    text = data["Country"])





traces = [trace1]



layout = dict(title="Relationship Between Population and Area Data",xaxis=dict(title="Population",ticklen=5),yaxis=dict(title="Area",ticklen=5))



fig = dict(data=traces,layout=layout)



normalized_df.head()
iplot(fig)
trace1 = go.Scatter(x=normalized_df["Population"],

                    y=normalized_df["Area (sq. mi.)"],

                    mode="lines",

                    name = "Population",

                    text = data["Country"])





traces = [trace1]



layout = dict(title="Relationship Between Population and Area Data",xaxis=dict(title="Population ",ticklen=5))



fig = dict(data=traces,layout=layout)



iplot(fig)
#Determining 10 most crowded country

crowded_countries_index = data["Population"].sort_values(ascending=False).index.values

crowded_countries = data.reindex(crowded_countries_index)

crd = crowded_countries.head(10)

crd
trace1 = go.Bar(x = crd.Country,

                y=crd.Population,

               name="Country Name",

               text = crd.Country,

               marker = dict(color="rgba(54,170,120,0.8)",line = dict(color="rgb(0,0,0)",width=1.5)))



trc_all = [trace1]



layout = go.Layout(dict(title="10 Most Populous Country in the World",xaxis = dict(title="Countries"),

                       yaxis=dict(title="Population")))



figure = go.Figure(data=trc_all,layout=layout)



iplot(figure)
#Histogram

trace1 = go.Histogram(x=data["Coastline (coast/area ratio)"]

                     ,name="Coastline Ratio"

                     ,marker=(dict(color="rgba(234,12,212,0.8)"))

                     ,opacity=0.7)



layout = go.Layout(title="Coastline Histogram"

                  ,xaxis=dict(title="Coastline")

                  ,yaxis=dict(title="Counts"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
coast_line_index = data["Coastline (coast/area ratio)"].sort_values(ascending=False).index.values

coastline_data = data.reindex(coast_line_index)

cst = coastline_data.head(10)

cst
#Bar plot



trace1 = go.Bar(x=cst.Country

      ,y=cst["Coastline (coast/area ratio)"]

      ,name="Coastline"

      ,text=cst.Region

      ,marker=dict(color="rgba(122,123,43,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



traces = [trace1]



layout = go.Layout(dict(title="Top 10 Countries of Coastline"),xaxis=dict(title="Countries"),yaxis=dict(title="Coastline Ratio"))



fig = go.Figure(data=traces,layout=layout)



iplot(fig)
trace = go.Pie(values=cst["Coastline (coast/area ratio)"]

              ,labels=cst.Country

              ,name="Countries"

              ,hoverinfo="label+percent+name")



layout = go.Layout(dict(title="10 Countries with the Most Coastline"))



figure = go.Figure(data=trace,layout=layout)



iplot(figure)
#Histogram

trace1 = go.Histogram(x=data["Net migration"]

                     ,name="Net Migration"

                     ,marker=(dict(color="rgba(234,231,43,0.8)"))

                     ,opacity=0.7)



layout = go.Layout(title="Net Migration Histogram"

                  ,xaxis=dict(title="Net Migration")

                  ,yaxis=dict(title="Counts"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Creating dataframes

positive_mig = data[data["Net migration"]>0] 

negative_mig = data[data["Net migration"]<0]



#Sorting values

new_index = positive_mig["Net migration"].sort_values(ascending=False).index.values

positive_mig = positive_mig.reindex(new_index)



new_index = negative_mig["Net migration"].sort_values().index.values

negative_mig = negative_mig.reindex(new_index)



#Taking a look at negative data

positive_mig.head()
#taking a look at negative data

negative_mig.head()
trace1 = go.Bar(x=positive_mig["Country"]

               ,y=positive_mig["Net migration"]

               ,name="Countries/Net Migration (Positive)"

               ,text=positive_mig.Region

               ,marker=dict(color="rgba(132,42,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))





trace2 = go.Bar(x=negative_mig["Country"]

               ,y=negative_mig["Net migration"]

               ,name="Countries/Net Migration (Negative)"

               ,text=negative_mig.Region

               ,marker=dict(color="rgba(42,230,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



layout = go.Layout(dict(title="Top 10 Countries of Datasets"

                        ,xaxis=dict(title="Countries")

                        ,yaxis=dict(title="Net Migration")))





figure = go.Figure(data=[trace1,trace2],layout=layout)



iplot(figure)
#Histogram

trace1 = go.Histogram(x=data["Infant mortality (per 1000 births)"]

                     ,name="Infant Mortality"

                     ,marker=(dict(color="rgba(0,238,90,1)"))

                     ,opacity=0.7)



layout = go.Layout(title="Infant Mortality Histogram"

                  ,xaxis=dict(title="Infant Mortality")

                  ,yaxis=dict(title="Counts"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
# Scatter Plot using Infant Mortality - GDP Per Capita



trace1 = go.Scatter(x=data["Infant mortality (per 1000 births)"]

                   ,y=data["GDP ($ per capita)"]

                   ,mode="markers"

                   ,name="Infant Mortality - GDP Per Capita"

                   ,text=data.Country)



layout = go.Layout(dict(title="Infant Mortality - GDP Per Capita"

                        ,xaxis=dict(title="Infant Mortality")

                        ,yaxis=dict(title="GDP Per Capita")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Infant Mortality - Literacy Scatter Plot

trace1 = go.Scatter(x=data["Infant mortality (per 1000 births)"]

                   ,y=data["Literacy (%)"]

                   ,mode="markers"

                   ,name="Infant Mortality - Literacy"

                   ,text=data.Country)



layout = go.Layout(dict(title="Infant Mortality - Literacy"

                        ,xaxis=dict(title="Infant Mortality")

                        ,yaxis=dict(title="Literacy")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Infant Mortality - Phones (per 1000)

trace1 = go.Scatter(x=data["Infant mortality (per 1000 births)"]

                   ,y=data["Phones (per 1000)"]

                   ,mode="markers"

                   ,name="Infant Mortality - Phones"

                   ,text=data.Country)



layout = go.Layout(dict(title="Infant Mortality - Phones"

                        ,xaxis=dict(title="Infant Mortality")

                        ,yaxis=dict(title="Phones")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Sorting values

sorted_indexes = data["Infant mortality (per 1000 births)"].sort_values(ascending=False).index.values

infant_df = data.reindex(sorted_indexes)



top5 = infant_df.tail()

last5 = infant_df.head()



#Taking a look at top5 data

top5
#taking a look at last5 data

last5
trace1 = go.Bar(x=top5.Country

               ,y=top5["Infant mortality (per 1000 births)"]

               ,name="Top 5 Countries by Infant Mortality"

               ,text=top5.Region

               ,marker=dict(color="rgba(213,21,90,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



trace2 = go.Bar(x=last5.Country

               ,y=last5["Infant mortality (per 1000 births)"]

               ,name="Last 5 Countries by Infant Mortality"

               ,text=last5.Region

               ,marker=dict(color="rgba(32,98,202,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



traces = [trace1,trace2]



layout = go.Layout(title="Infant Mortality Bar Plot"

                  ,xaxis=dict(title="Country Names")

                  ,yaxis=dict(title="Infant Mortality"))



figure = go.Figure(data=traces,layout=layout)



iplot(figure)
#Concating dataframes

t5_nd_l5 = pd.concat([top5,last5],axis=0)

t5_nd_l5
trace1 = go.Pie(values=t5_nd_l5["Infant mortality (per 1000 births)"]

               ,labels=t5_nd_l5.Country

               ,name="Infant Mortality per 1000 births"

               ,hoverinfo= "label+percent+name")





layout = go.Layout(dict(title="Top 5 and Last 5 countries by Infant Mortality"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Histogram

trace1 = go.Histogram(x=data["GDP ($ per capita)"]

                     ,name="GDP per capita"

                     ,marker=(dict(color="rgba(181,118,54,1)"))

                     ,opacity=0.9)



layout = go.Layout(title="GDP per capita Histogram"

                  ,xaxis=dict(title="GDP Per capita")

                  ,yaxis=dict(title="Counts"))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#GDP - Infant Mortality (-0.6) Visualization



trace1 = go.Scatter(x=data["Infant mortality (per 1000 births)"]

                   ,y=data["GDP ($ per capita)"]

                   ,name="Infant Mortality -  GDP Per Capita ",

                    mode = "markers"

                   ,text=data.Country

                   ,marker =dict(color="rgba(68,234,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



layout = go.Layout(dict(title="Infant Mortality - GDP Per Capita Visualization",

                        xaxis=dict(title="Infant Mortality"),

                        yaxis=dict(title="GDP Per Capita")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#GDP - Literacy (0.5) Visualization



trace1 = go.Scatter(x=data["Literacy (%)"]

                   ,y=data["GDP ($ per capita)"]

                   ,name="Literacy -  GDP Per Capita ",

                    mode = "markers"

                   ,text=data.Country

                   ,marker =dict(color="rgba(68,234,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



layout = go.Layout(dict(title="Literacy - GDP Per Capita Visualization",

                        xaxis=dict(title="Literacy"),

                        yaxis=dict(title="GDP Per Capita")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#GDP - Phones (0.88) Visualization



trace1 = go.Scatter(x=data["Phones (per 1000)"]

                   ,y=data["GDP ($ per capita)"]

                   ,name="Phones -  GDP Per Capita ",

                    mode = "markers"

                   ,text=data.Country

                   ,marker =dict(color="rgba(68,234,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



layout = go.Layout(dict(title="Phones (per 1000) - GDP Per Capita Visualization",

                        xaxis=dict(title="Phones"),

                        yaxis=dict(title="GDP Per Capita")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#GDP -Birthrate (-0.65) Visualization



trace1 = go.Scatter(x=data["Birthrate"]

                   ,y=data["GDP ($ per capita)"]

                   ,name="Birthrate -  GDP Per Capita ",

                    mode = "markers"

                   ,text=data.Country

                   ,marker =dict(color="rgba(68,234,187,0.7)",line=dict(color="rgb(0,0,0)",width=1.5)))



layout = go.Layout(dict(title="Birthrate - GDP Per Capita Visualization",

                        xaxis=dict(title="Birthrate"),

                        yaxis=dict(title="GDP Per Capita")))



figure = go.Figure(data=trace1,layout=layout)



iplot(figure)
#Determining 

top10_index = data["GDP ($ per capita)"].sort_values(ascending=False).index.values

last10_index = data["GDP ($ per capita)"].sort_values().index.values

top10_gdp = data.reindex(top10_index)

last10_gdp = data.reindex(last10_index)

top10_gdp= top10_gdp.iloc[:10]

last10_gdp = last10_gdp.iloc[:10]

top10_gdp
last10_gdp
#Visualizing

trace1 = go.Bar(x=top10_gdp["Country"]

               ,y=top10_gdp["GDP ($ per capita)"]

               ,name="Top 10 Countries by GDP per capita"

               ,text=top10_gdp.Region

               ,marker=(dict(color="rgba(213,52,5,0.8)",line=dict(color="rgb(0,0,0)",width=1.5))))





trace2 = go.Bar(x=last10_gdp["Country"]

               ,y=last10_gdp["GDP ($ per capita)"]

               ,name="Last 10 Countries by GDP per capita"

               ,text=last10_gdp.Region

               ,marker=(dict(color="rgba(94,21,231,0.8)",line=dict(color="rgb(0,0,0)",width=1.5))))



layout = go.Layout(dict(title="Top and Last 10 Countries by GDP per capita")

                  ,xaxis=dict(title="Country"),yaxis=dict(title="GDP per capita"))



figure = go.Figure(data=[trace1,trace2],layout=layout)



iplot(figure)
#function creating

def boxplotCreator(dataFrame,yValueList,colorList,nameList):

    traces = []

    for yValue,color,name in (zip(yValueList,colorList,nameList)):

        trace1 = go.Box(y=dataFrame[yValue]

                       ,marker=(dict(color=color))

                       ,name=name)

        

        traces.append(trace1)

        

    

    figure = go.Figure(data=traces)

    iplot(figure)
numerical_features = [i for i in data.corr().columns]

numftr1 = numerical_features[:3]

numftr2 = numerical_features[3:6]

numftr3 = numerical_features[6:9]

numftr4 = numerical_features[9:]

colors = ["rgba(89,232,84,0.9)"

         ,"rgba(147,242,88,0.8)"

         ,"rgba(67,237,55,0.9)"

         ,"rgba(141,40,252,0.8)"

         ,"rgba(40,166,252,0.8)"

         ,"rgba(111,253,182,0.7)"

         ,"rgba(27,143,212,0.6)"

         ,"rgba(123,198,219,0.8)"

         ,"rgba(129,21,124,0.7)"

         ,"rgba(142,12,43,0.8)"

         ,"rgba(5,145,239,0.8)"

         ,"rgba(43,35,31,0.8)"]

colors1 = colors[:3]

colors2 = colors[3:6]

colors3 = colors[6:9]

colors4 = colors[9:]



boxplotCreator(data,numftr1,colors1,numftr1)
boxplotCreator(data,numftr2,colors2,numftr2)
boxplotCreator(data,numftr3,colors3,numftr3)
boxplotCreator(data,numftr4,colors4,numftr4)
country_name = ["United States of America","United Kingdom","Germany",

               "Greece","Turkey","Norway","South Africa","Netherlands",

               "Japan","Canada"]



country_lat = [38,54,51,39,

              39,62,-29,52.30,36,60]



country_lon = [-97,-2,9,22,35,10,24,5.45,138,-95]



country_locations = pd.DataFrame(dict(Country=country_name,CountryLat=country_lat,CountryLon=country_lon))



country_locations
#trace

trace1 = go.Scattergeo(lon=country_locations.CountryLon

                      ,lat=country_locations.CountryLat

                      ,text=country_locations.Country

                      ,name="text"

                      ,mode="markers")





layout = go.Layout(dict(title="Some Countries of the World"

                       ,hovermode="closest"

                       ,geo = dict(showframe=False,showland=True,showcoastlines=True,showcountries=True)))





iplot(go.Figure(data=trace1,layout=layout))
#I am preparing my data for use

GDP_index = data["GDP ($ per capita)"].sort_values(ascending=False).index.values

sorted_data = data.reindex(GDP_index)

sorted_data = sorted_data.head(20)

sorted_data.head()
cols = [i for i in data.corr().columns

        if not i== "Population" 

        and not i=="Area (sq. mi.)" and not i=="GDP ($ per capita)"

        and not i=="Phones (per 1000)"] # columns



fig,ax = plt.subplots(figsize=(12,10))

pd.plotting.parallel_coordinates(sorted_data, class_column='Country', cols=cols,colormap=plt.get_cmap("plasma"),linewidth=3)

plt.xlabel("Features")

plt.title("Parallel Cordinates Plot",fontsize=15)

plt.xticks(rotation=90)

plt.show()
data.corr()
#3D Scatter



trace = go.Scatter3d(x=data["Phones (per 1000)"]

                    ,y=data["GDP ($ per capita)"]

                    ,z=data["Literacy (%)"]

                    ,mode="markers"

                    ,marker=dict(color="rgba(42,153,34,0.7)")

                    ,name="Phones | GDP | Literacy"

                    ,text=data.Country)



layout = go.Layout(dict(title="Comparing Three Feauteres - Phones | GDP | Literacy"))

figure = go.Figure(data=trace,layout=layout)



iplot(figure)