# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.plotly as py  # plotly
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

from wordcloud import WordCloud  # word cloud library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# READ data
data=pd.read_csv("../input/vgsales.csv")
data.info()
display(data.head())
display(data.tail())
#Rank, name, platform, genre, all sales don't have missign value. 
#Column of Year has 271 missign values and column of publisher has 58 missign values.
# import missingno library
import missingno as msno
msno.bar(data)
plt.show()
#count of Null values
print(data.isnull().sum())

#Remove missing values.
data = data.dropna()
print('---')
data.Year=data.Year.astype(int)
data.info()
print('Genre of Games:',data.Genre.unique())
print('Unique Platform Count:',len(data.Platform.unique()))
#Average Global Sales on the Genre of Games
genre=list(data.Genre.unique())
global_sales=[]
for i in genre:
    val=data[data.Genre==i]
    x=val.Global_Sales.mean()
    global_sales.append(x)
    
d1 = pd.DataFrame({"Genre":genre,"Global_Sales":global_sales})
d1.sort_values("Global_Sales",ascending=False,inplace=True)

#visualization
plt.figure(figsize=(12,7))
sns.barplot(x="Genre", y="Global_Sales", data=d1, palette="RdBu")
#sns.despine(left = True, right = True)
plt.xticks(rotation= 30)
plt.xlabel("Genre", fontsize=16, color="blue")
plt.ylabel("Global_Sales", fontsize=16, color="blue")
plt.title("Global Sales on the Genre of Games", fontsize=18, color="red")
plt.show()
# Average Global Sales on the Platform
platform=list(data.Platform.unique())
global_sales=[]
for i in platform:
    val=data[data.Platform==i]
    x=val.Global_Sales.mean()
    global_sales.append(x)
      
d2=pd.DataFrame({'Platform':platform,'Global_Sales': global_sales})
d2.sort_values("Global_Sales",ascending=True,inplace=True)

#visualization
plt.figure(figsize=(12,8))
sns.barplot(x="Platform", y="Global_Sales", data=d2, palette="deep")
#sns.despine(left = True, right = True)
plt.xticks(rotation= 45)
plt.xlabel("Platform", fontsize=16, color="#123456")
plt.ylabel("Global_Sales", fontsize=16, color="#123456")
plt.title("Global Sales on the Platform", fontsize=18, color="#dd5522")
plt.show()
# Rate of sales according to Genre of Games that are North America, Euro, Japan, Other and Global Sales.
genre_list=list(data.Genre.unique())
na_sales=[]
eu_sales=[]
jp_sales=[]
other_sales=[]
global_sales=[]
for i in genre_list:
    val=data[data.Genre==i]
    na_sales.append(val.NA_Sales.mean())
    eu_sales.append(val.EU_Sales.mean())
    jp_sales.append(val.JP_Sales.mean())
    other_sales.append(val.Other_Sales.mean())
    global_sales.append(val.Global_Sales.mean())
    
f,ax = plt.subplots(figsize = (12,7))
sns.barplot(x=na_sales,y=genre_list,color='orange',alpha = 0.8,label='NA_Sales' )
sns.barplot(x=eu_sales,y=genre_list,color='blue',alpha = 0.7,label='EU_Sales')
sns.barplot(x=jp_sales,y=genre_list,color='red',alpha = 0.7,label='JP_Sales')
sns.barplot(x=other_sales,y=genre_list,color='green',alpha = 0.6,label='Other_Sales')
sns.barplot(x=global_sales,y=genre_list,color='grey',alpha = 0.5,label='Global_Sales')

ax.legend(loc='lower right',frameon = True)
ax.set(xlabel='Sales', ylabel='Genre',title = "Rate of Sales  According to Genre of Games")

#Correlation
data.corr()
f,ax = plt.subplots(figsize=(6, 6))
sns.heatmap(data.corr(), annot=True, linewidths=.9, linecolor='black', fmt= '.2f',ax=ax)
plt.show()
#Global sales over the years.
sns.jointplot(data.Year,data.Global_Sales,size=8, ratio=9, color="blue")
plt.show()
data.Genre.value_counts()
#Games according to Genre
labels=data.Genre.value_counts().index
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = data.Genre.value_counts().values
# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=sns.color_palette('Set2'), autopct='%1.1f%%')
plt.title('Games According to Genre',fontsize = 17,color = 'green')
# Count of Platforms
plt.subplots(figsize = (15,8))
sns.countplot(data.Platform)
plt.title("Platform",color = 'blue',fontsize=20)
#World rank of the top 100 video games, north america sales and europe sales.
df=data.loc[:99,:] # data.iloc[:100,:] -- data.head(100)

import plotly.graph_objs as go

trace1=go.Scatter(
                x=df.Rank,
                y=df.NA_Sales,
                mode="lines+markers",
                name="North America Sales",
                marker= dict(color = 'rgba(158, 90, 10, 0.7)'),
                text=df.Name)
trace2=go.Scatter(
                x=df.Rank,
                y=df.EU_Sales,
                mode="lines",
                name="Europe Sales",
                marker=dict(color = 'rgba(56, 140, 200, 0.7)'),
                text=df.Name)

edit_df=[trace1,trace2]
layout=dict(title="World rank of the top 100 video games, north america and europe sales .",
            xaxis=dict(title="World Rank",tickwidth=5,ticklen=8,zeroline=False))
fig=dict(data=edit_df,layout=layout)
iplot(fig)
#The genres according to the average sales
# data preparation
dataGenre_list=list(data.Genre.unique())
na_sales=[]
eu_sales=[]
jp_sales=[]
other_sales=[]
global_sales=[]
for i in dataGenre_list:
    val=data[data.Genre==i]
    na_sales.append(val.NA_Sales.mean())
    eu_sales.append(val.EU_Sales.mean())
    jp_sales.append(val.JP_Sales.mean())
    other_sales.append(val.Other_Sales.mean())
    global_sales.append(val.Global_Sales.mean())
    
df1=pd.DataFrame({"Genre":dataGenre_list,"NA_Sales":na_sales,"EU_Sales":eu_sales,"JP_Sales":jp_sales,"Other_Sales":other_sales,"Global_Sales":global_sales})

# data visualization
trace1=go.Scatter(
                x=df1.Genre,
                y=df1.NA_Sales,
                mode="lines+markers",
                name="North America Sales",
                marker= dict(color = 'rgba(25, 25, 255, 0.7)'))
trace2=go.Scatter(
                x=df1.Genre,
                y=df1.EU_Sales,
                mode="lines",
                name="Europe Sales",
                marker=dict(color = 'rgba(25, 255, 25, 0.7)'))
trace3=go.Scatter(
                x=df1.Genre,
                y=df1.JP_Sales,
                mode="lines+markers",
                name="Japan Sales",
                marker=dict(color = 'rgba(255, 25, 25, 0.7)'))
trace4=go.Scatter(
                x=df1.Genre,
                y=df1.Other_Sales,
                mode="lines",
                name="Other Sales",
                marker=dict(color = 'rgba(25, 25, 25, 0.7)'))
trace5=go.Scatter(
                x=df1.Genre,
                y=df1.Global_Sales,
                mode="lines+markers",
                name="Global Sales",
                marker=dict(color = 'rgba(250, 150, 0, 0.7)'))
edit_df=[trace1,trace2,trace3,trace4,trace5]
layout=dict(title="The genres according to the average sales",
            xaxis=dict(title="  Genre of Video Games",tickwidth=5,ticklen=8,zeroline=False))
fig=dict(data=edit_df,layout=layout)
plt.savefig('graph.png')
iplot(fig)
# Sales of the top 4 Video Games in 2016
df2016=data[data.Year==2016].iloc[:4,:]

import plotly.graph_objs as go

trace1=go.Bar(
                x=df2016.Name,
                y=df2016.NA_Sales,
                name="NA_Sales",
                marker= dict(color = 'rgba(158, 90, 10, 0.7)',
                            line=dict(color='rgb(0,0,0)',width=1.9)),
               text=df2016.Publisher)
trace2=go.Bar(
                x=df2016.Name,
                y=df2016.EU_Sales,
                name="EU_Sales",
                marker=dict(color = 'rgba(56, 140, 200, 0.7)',
                           line=dict(color='rgb(0,0,0)',width=1.9)),
                text=df2016.Publisher)
trace3=go.Bar(
                x=df2016.Name,
                y=df2016.JP_Sales,
                name="JP_Sales",
                marker=dict(color = 'rgba(156, 30, 130, 0.7)',
                           line=dict(color='rgb(0,0,0)',width=1.9)),
                text=df2016.Publisher)
trace4=go.Bar(
                x=df2016.Name,
                y=df2016.Other_Sales,
                name="Other_Sales",
                marker=dict(color = 'rgba(100, 240,50 , 0.7)',
                           line=dict(color='rgb(0,0,0)',width=1.9)),
                text=df2016.Publisher)

edit_df=[trace1,trace2,trace3,trace4]
layout=go.Layout(barmode="group",title='Sales of the top 4 Video Games in 2016')
fig=dict(data=edit_df,layout=layout)
#fig = go.Figure(data = edit_df, layout = layout)
plt.savefig('graph.png')
iplot(fig)
#Rates of Video Games Global Sales in 2016
df2016=data[data.Year==2016].iloc[:10,:]
pie_list=list(df2016.Global_Sales)
labels=df2016.Publisher
fig={
    "data":[
        {
            "values":pie_list,
            "labels":labels,
            "domain": {"x": [.2, 1]},
            "name": "Rates of Video Games ",
            "hoverinfo":"label+percent+name",
            "hole": .4,
            "type": "pie"
        },],
    "layout":{
        "title":"Rates of Video Games Global Sales in 2016",
        "annotations":[
            {
                "font":{"size":17},
                "showarrow": False,
                "text": "Video Games",
                "x": 0.75,
                "y": 0.5
            },
        ]
    }  
}
iplot(fig)
# Which video games is mentioned most at Genre of Action..
# data prepararion
w_data = data[data.Genre == 'Action'].iloc[:50,:]
plt.subplots(figsize=(8,8))
wordcloud = WordCloud(
                        background_color='#f2f2f2',
                        width=532,
                        height=374
                     ).generate(" ".join(w_data.Name))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')
plt.show()
# World rank of the top 100 video games, japan sales and europe sales.
dframe=data.iloc[:100,:]
trace0=go.Scatter(
    x=dframe.Rank,
    y=dframe.JP_Sales,
    name = "Japan Sales",
    marker = dict(color = 'rgba(56, 182, 32, 0.8)'),
)
trace1=go.Scatter(
    x=dframe.Rank,
    y=dframe.EU_Sales,
    xaxis='x2',
    yaxis='y2',
    name = "Europe Sales",
     marker = dict(color = 'rgba(196, 20, 92, 0.8)'),
)
edit_data=[trace0,trace1]
layout = go.Layout(
    xaxis2=dict(
        domain=[0.5, 0.95],
        anchor='y2',        
    ),
    yaxis2=dict(
        domain=[0.5, 0.95],
        anchor='x2',
    ),
    title = ' World rank of the top 100 video games, japan sales and europe sales.'
)
fig = go.Figure(data=edit_data, layout=layout)
iplot(fig)

# World rank of the top 100 video games, japan sales and europe sales.
trace1=go.Scatter3d(
    x=dframe.Rank,
    y=dframe.EU_Sales,
    z=dframe.JP_Sales,
    mode='markers',
    marker=dict(
        size=12,
        #color='rgb(135,25,25)',
        color=dframe.Rank,                # set color to an array/list of desired values
        colorscale='Viridis',             # choose a colorscale
        opacity=0.9
    )
)
edit_data=[trace1]
layout=go.Layout(
     margin=dict(
        l=1,
        r=1,
        b=1,
        t=1  
    )
)
fig=go.Figure(data=edit_data,layout=layout)
iplot(fig)
# World rank of top 30 Video Games Sales data class visualization according to features
from pandas.tools.plotting import parallel_coordinates

df=data.drop(['Rank','Name','Year','Platform','Publisher','Global_Sales'],axis=1)

plt.figure(figsize=(15,8))
parallel_coordinates(df.head(30), 'Genre', colormap=plt.get_cmap("tab10"))
plt.title("Video Games Sales data class visualization according to features (NA_Sales, EU_Sales, JP_Sales, Other_Sales)")
plt.xlabel("Features of data set")
plt.ylabel("sales")
plt.show()
# correlation
corr=df.corr()
corr
# We seen the correlation between north america, europe , japan and other sales.
# import networkx library
import plotly.graph_objs as go
import pandas as pd
import networkx as nx


links = corr.stack().reset_index()
links.columns = ['column1', 'column2','value']

# correlation
threshold = 0

links_filtered=links.loc[ (links['value'] >= threshold ) & (links['column1'] != links['column2']) ] 
G=nx.from_pandas_edgelist(links_filtered, 'column1', 'column2')
nx.draw_circular(G, with_labels=True, node_color='cyan', node_size=320, edge_color='green', linewidths=2, font_size=10)