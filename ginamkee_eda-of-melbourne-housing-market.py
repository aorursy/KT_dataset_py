import numpy as np

import pandas as pd

import os

import seaborn as sns

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/melbourne-housing-market/Melbourne_housing_FULL.csv')
df.info()
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
df = df.drop([29483,18443,25717,27390,27391,2536,26210,12043,12043,27150,6017,25839,12539,19583,25635])
df.loc[18523,'Regionname']='Western Metropolitan'

df.loc[26888,'Regionname']='Southern Metropolitan'
df.loc[18523,'Propertycount']=7570.0

df.loc[26888,'Propertycount']=8920.0
fre = df['Bedroom2'].mode()[0]

df['Bedroom2'] = df['Bedroom2'].fillna(fre)
fre = df['Bathroom'].mode()[0]

df['Bathroom'] = df['Bathroom'].fillna(fre)
df['Regionname'] = df['Regionname'].map({'Southern Metropolitan':0, 'Northern Metropolitan':1, 'Western Metropolitan':2, 'Eastern Metropolitan':3

                                        , 'South-Eastern Metropolitan':4, 'Eastern Victoria':7, 'Northern Victoria':6, 'Western Victoria':5}).astype(int)
df = df.drop(['SellerG', 'CouncilArea', 'Date', 'BuildingArea', 'YearBuilt', 'Landsize', 'Car'], axis=1)
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
fre = df['Lattitude'].mode()[0]

df['Lattitude'] = df['Lattitude'].fillna(fre)
fre = df['Longtitude'].mode()[0]

df['Longtitude'] = df['Longtitude'].fillna(fre)
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot



cnt_srs = df['Price'].value_counts()

trace1 = go.Scatter(

    x = cnt_srs.index,

    y = cnt_srs.values,

    mode = "markers",

    marker = dict(color = 'rgba(200, 50, 55, 0.8)')

)

data = [trace1]

layout = dict(title = 'Distribution of Price',

xaxis= dict(title= 'Price per house',ticklen= 5,zeroline= False)

)

fig = go.Figure(data = data, layout = layout)

fig
df['pricepie'] = np.where(df.Price > 1, '1','0')
cnt_ = df['pricepie'].value_counts()

fig = {

"data": [

{

"values": cnt_.values,

"labels": cnt_.index,

"domain": {"x": [0, .5]},

"name": "Train types",

"hoverinfo":"label+percent+name",

"hole": .7,

"type": "pie"

},],

"layout": {

"title":"NaN of Price ",

"annotations": [

{ "font": { "size": 20},

"showarrow": False,

"text": "Pie Chart",

"x": 0.50,

"y": 1

},

]

}

}

iplot(fig)
df['Price'] = df['Price'].fillna(method='ffill')

df.loc[0,'Price']=1480000.0
cnt_srs = df['Price'].value_counts()

trace1 = go.Scatter(

    x = cnt_srs.index,

    y = cnt_srs.values,

    mode = "markers",

    marker = dict(color = 'rgba(200, 50, 55, 0.8)')

)

data = [trace1]

layout = dict(title = 'Distribution of Price after missing value processing',

xaxis= dict(title= 'Price per house',ticklen= 5,zeroline= False)

)

fig = go.Figure(data = data, layout = layout)

fig
total = df.isnull().sum().sort_values(ascending=False)

percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])



percent_data = percent.head(20)

percent_data.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)

plt.ylabel("Count", fontsize = 20)

plt.title("Total Missing Value (%)", fontsize = 20)
corrMatrix=df[["Price","Rooms","Bedroom2","Bathroom",'Longtitude',"Postcode"]].corr()

sns.set(font_scale=1.10)

plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,

square=True,annot=True,cmap='viridis',linecolor="white")

plt.title('Correlation between features');
corr_matrix = df.corr()

corr_matrix["Price"].sort_values(ascending=False)
plt.title('Distribution of Region name')

df['Regionname'].value_counts().plot.bar(color=('r', 'g', 'c', 'b', 'y'))
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

        color=color,

        ),

    )

    return trace

cnt_srs = df.groupby('Regionname')['Price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(1000, 11, 20, 0.6)')

layout = go.Layout(title = 'Region name by Distance')

fig = go.Figure(data = trace0, layout = layout)

fig
def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

        color=color,

        ),

    )

    return trace

cnt_srs = df.groupby('Bathroom')['Price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(1000, 620, 100, 0.6)')

layout = go.Layout(title = 'Region name by Distance')

fig = go.Figure(data = trace0, layout = layout)

fig
ax = df.plot(kind="scatter", x="Longtitude", y="Lattitude", alpha=0.4,

    s=df["Propertycount"]/100, label="Propertycount", figsize=(10,7),

    c="Price", cmap=plt.get_cmap("jet"), colorbar=True,

    sharex=False)

ax.set(xlabel='longitude', ylabel='Latitude')

plt.legend()
import folium

from folium import plugins

map1 = folium.Map(location=[-37.8014, 144.9958], zoom_start=5)

markers = []

for n in df.index:

    folium.CircleMarker([df['Lattitude'][n], df['Longtitude'][n]], radius = df['Propertycount'][n]*0.00003, color='#ef4f61', fill=True).add_to(map1)

map1
dfs = df[(df["Regionname"] == 0)]
plt.title('Suburbs of Southern Metropolitan')

dfs['Suburb'].value_counts().plot.bar(color=('r', 'g', 'c', 'b', 'y'), figsize=(15, 10))
import plotly.graph_objs as go





def horizontal_bar_chart(cnt_srs, color):

    trace = go.Bar(

        y=cnt_srs.index[::-1],

        x=cnt_srs.values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace

cnt_srs = dfs.groupby('Suburb')['Price'].agg(['mean'])

cnt_srs.columns = ["mean"]

cnt_srs = cnt_srs.sort_values(by="mean", ascending=False)

trace0 = horizontal_bar_chart(cnt_srs['mean'], 'rgba(50, 71, 96, 0.6)')

layout = go.Layout(title = 'Average prices by Suburbs of Southern Metropolitan', width=1000, height=1000)

fig = go.Figure(data = trace0, layout = layout)

fig
dfs = dfs[(dfs["Type"]=="h") & (dfs["Price"] >= 4000000) & (dfs["Suburb"] == 'Canterbury')]

dfs