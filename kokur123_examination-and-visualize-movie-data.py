# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

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
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data.columns
data.info()
del data["homepage"]
del data["id"]
del data["keywords"]
del data["overview"]
del data["production_companies"]
del data["spoken_languages"]
del data["status"]
del data["tagline"]
del data["title"]
data.columns
data.head()
import json 

data["genres"] = data["genres"].apply(json.loads)
data["production_countries"]= data["production_countries"].apply(json.loads)


def pipe_flatten_names(keywords):
    return '|'.join([x['name'] for x in keywords])


data['genres'] = data['genres'].apply(pipe_flatten_names)
data["production_countries"]=data["production_countries"].apply(pipe_flatten_names)



data.loc[:8,["original_title","genres","production_countries"]] #Let's see if it's arrangeddata.
data.index.name="index"
data.tail()
data.describe()
data.dtypes #release_date is object 
data.loc[:20,"original_title"] #First twenty movie
df = data.head(7) 
melted = pd.melt(frame=df,id_vars = "original_title",value_vars=["genres"]) #we melted the df 
melted
melted.pivot(index="original_title",columns= "variable",values = "value") #We returned the melt data
overall_average_rating= sum(data.vote_average)/len(data.original_title) # the average number of movies and total number of films 
data["vote_level"] = [ "high_level" if each>overall_average_rating else "down_level"  for each in data.vote_average] #We have defined a new column according to the average vote.

data.loc[:5,["original_title","vote_level","vote_average"]] #let's  see that
data3 = data[data.vote_average > 9] # we may find top 6 movie
data3.loc[:,["original_title","vote_average"]] #let's see top 6 movie 
data.original_language.unique() #see the tongues of films
data.original_language.value_counts(dropna = False) #let's look at the frequency of the languages of the films
c=pd.Series(["2" if each == 0 else 1 for each in data.budget]) #See the invalid construction fees
c.value_counts()
average_cost_of_living = int(sum(data.budget)/3766)
data.budget.replace([0],average_cost_of_living,inplace=True) #In our graphics, we are editing the data so that it does not cause anomalies.
data.tail()
d=pd.Series(["2" if each == 0 else 1 for each in data.revenue])
d.value_counts()
average = int(sum(data.revenue)/3376)
data.revenue.replace([0], average, inplace = True)
data.tail()
#I will draw a graphic that shows film popularity according to languages. We need to get the first data ready for this.
#Seaborn
df=data.copy()

unique = list(df.original_language.unique())
list_ratio=[]
for each in unique:
    x= df[df["original_language"] == each]
    ratio_popularity=sum(x.popularity)/len(x)
    list_ratio.append(ratio_popularity)
    
df2 = pd.DataFrame({"language":unique,"ratio":list_ratio})
new_index = (df2.ratio.sort_values(ascending = False)).index.values
sorted_data= df2.reindex(new_index)

#Visualization
plt.figure(figsize = (20,12))
sns.barplot(x= sorted_data["language"],y  = sorted_data["ratio"])
plt.xticks(rotation= 90)
plt.xlabel("Language",fontsize=15)
plt.ylabel("Popularity",fontsize= 15)
plt.title("Language and Filmin popularity",fontsize= 20)
sorted_data.head(7)
#Seaborn
from collections import Counter

df = data.genres.copy()

list_kind = df.str.split("|")
a = []
for each in list_kind:
    for i in each:
        a.append(i)
        
c=[]
for each in a:
    if each != "":
         c.append(each)        
        
f= dict(Counter(c))

df3 = pd.DataFrame(list(f.items()),columns = ["kind","ratio"])
new_index =( df3.ratio.sort_values(ascending=False)).index.values
new = df3.reindex(new_index)



plt.figure( figsize = (15,10))
sns.barplot(x="kind",y="ratio",data=new,palette = sns.cubehelix_palette(len(x)))
plt.xticks(rotation = 90)
plt.xlabel("Kind Of Movie",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.title("Number of movie types",fontsize = 20)

new.head(7)
#Seaborn
plt.figure(figsize = (17,10))
   
sns.barplot(x = "budget", y = "original_title",data= data.head(10),color= "red", alpha=0.5,label ="Budget")
sns.barplot(x = "revenue",y= "original_title", data=  data.head(10),color="green",alpha=0.5,Label="Revenue")

plt.text(2500000000,8.5,"Revenue",color="green",fontsize = 17 ,style ="italic")
plt.text(2500000000,9,"Budget", color="red",  fontsize = 17 ,style ="italic")

plt.xlabel("İncome And Expensive",fontsize= 15)
plt.ylabel("Movie",fontsize= 15)
plt.title("The First 10 Movie income and expense",fontsize = 20)

#Plotly
trace1 = go.Bar(
    x = sorted_data.language,
    y = sorted_data.ratio,
    name = "Ratio",
    marker = dict(
        color = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
        colorscale = "Bluered")
)

data1= [trace1]
layout = dict( 
    autosize = False,
    width = 1378,
    height = 720,
    barmode = "group")
fig  = dict (data = data1, layout = layout)
iplot ( fig)
#Seaborn
f,ax1 =plt.subplots(figsize = (50,20))
                    
sns.pointplot(x ="original_title",y = "runtime",    data= data.head(50),color = "red")
sns.pointplot(x ="original_title",y = "popularity", data= data.head(50), color = "green")

plt.xticks(rotation = 85,fontsize = 25)
plt.yticks(fontsize=25)

ax1.text(45,400,"Runtime",color ="red",fontsize= 35 , style ="italic")
ax1.text(45,385,"Popularity",color = "green",fontsize = 35,style ="italic")

ax1.set_xlabel("Original Title",    fontsize = 30,color="blue")
ax1.set_ylabel("Runtime and Popularity",fontsize=30,color="blue")
ax1.set_title("Runtime vs Popularity",fontsize = 40,color="blue")
   
  

plt.grid()

#Plotly
import plotly.graph_objs as go 

df = data.head(100).copy()

trace1 =go.Scatter(
    x =df.index,
    y = df.popularity,
    mode ="lines",
    name = " Popularity",
    marker = dict(color = "rgb(242, 99, 74,0.7)"),
    text = df.original_title
)
trace2 = go.Scatter(
    x = df.index,
    y = df.runtime,
    mode = "lines + markers",
    name = "Runtime",
    marker = dict( color = "rgb(144, 211, 74,0.5)"),
    text = df.original_title
)
trace3 = go.Scatter(
    x = df.index,
    y = df.vote_average,
    mode = "markers",
    name = "Vote Averge",
    marker = dict(color = "rgb(118, 144, 165)"),
    text = df.original_title
)
data1=[trace1,trace2,trace3]
layout = dict(
    title = "Runtime vs Popularity"
)
fig = dict ( data = data1 , layout = layout)
iplot(fig)
#Seaborn
plt.figure( figsize = (15,5))
sns.regplot( data.vote_average.head(200),data.vote_count.head(200), color = "g" )
plt.show()
#Plotly
df = data.head(200).copy()

trace1 = go.Scatter(
    x = df.index,
    y = df.vote_average,
    mode = "markers",
    name = "Vote Average",
    marker =dict( color = "rgb(120, 171, 200,0)"),
    text = df.original_title
)
trace2 = go.Scatter(
    x = df.index,
    y = df.vote_count,
    mode ="markers",
    name = " Vote Count",
    marker =dict (
        color = "rgb(168, 229, 183)",
        size = 10,
        line = dict(
            color = "rgb(251, 203, 251)",
            width = 2
        )
    ),
    text = df.original_title
)
data1 = [trace1,trace2]
layout = dict( title = " Vote Average and Vote Count")
fig = dict ( data = data1 , layout = layout)
iplot( fig)
#Seaborn
sns.jointplot(data.vote_average.head(100),data.vote_count.head(100),kind ="reg",size=8,color="grey")
plt.show()
#Seaborn
sns.jointplot(data.vote_average.head(100),data.vote_count.head(100),kind ="kde",size=8,color="g")
plt.show()
#Seaborn
df= data.loc[:,["vote_count","popularity"]]
sns.pairplot(df)
plt.show()
#Plotly
import plotly.figure_factory as ff
data1 = data.loc[:,["vote_average","vote_count"]]
data1["index"] = np.arange(1,len(data1)+1)

fig = ff.create_scatterplotmatrix(data1,diag= "box", index = "index",colormap = "Portland",colormap_type = "cat",
                                 height = 800,width=1200)
iplot(fig)
data1.head(7)
#Matplotlib(I could not find a pie plot drawn with Seaborn.)
a=[]
for each in data.production_countries.str.split("|"):
    for i in each:
        a.append(i)
        
b = dict(Counter(a))

keys=[]
values=[]

for key,value in b.items() :
    if value > 30 and key != "":
        keys.append(key)
        values.append(value)

       
labels = keys
colors = sns.color_palette()
explode =[0,0,0,0,0,0,0,0,0,0,0,0,0]
sizes= values
    
plt.figure(figsize = (15,15))

plt.pie(sizes,explode = explode,labels=labels,colors = colors,autopct='%1.1f%%',textprops= {"fontsize": 10},shadow = False)

plt.show()
keys
values
#Plotly
trace1 = go.Pie(
    labels = keys,
    values = values,
    name = "Movie Percent",
    hoverinfo = "label+percent+name",
    domain = dict ( x = [0,1]),
    hole = .2
)
data1 = [trace1]
layout = dict(title = "Film Production Rates According To Countries")
fig = dict ( data = data1,layout= layout)
iplot(fig)
#Seaborn
f,ax = plt.subplots(figsize = (10,10))
sns.heatmap (data.corr(), annot = True,linewidths =0.75,linecolor = "White",fmt = ".2f",ax = ax,center = -0.1)
plt.show()
data.corr()
#Plotly(I just wanted to show you that you can draw)
trace1=go.Heatmap(
    x=["vote_count","vote_average","runtime"],
    y=["vote_count","vote_average"],
    z=[[1.00,0.31,0.27],[0.31,1.00,0.38]],
    colorscale= "viridis"
)
iplot([trace1])
#Plotly
trace1= go.Box(
    x =data.vote_average,
    name = "Vote Average",
    marker = dict ( color = "#666699")
)

iplot([trace1])
#Seaborn
plt.figure(figsize = (20,9))
sns.boxplot(x = "original_language", y ="vote_average",hue = "vote_level",data = data)

plt.xlabel ("Original language", fontsize = 20,color = "red")
plt.ylabel ("Vote average", fontsize = 20,color = "green")
plt.xticks( rotation = 45,fontsize = 15)
plt.yticks( fontsize = 15)
plt.show()
#Seaborn
plt.figure( figsize = (15,9))
sns.swarmplot(x = "original_language", y ="vote_average",hue = "vote_level",data = data.head(1000))
plt.show()
#Seaborn
df = data.loc[:,["runtime","popularity"]].copy()
plt.figure( figsize = (10,10))
sns.violinplot(data=df , palette = sns.cubehelix_palette(dark = 0.8,light = 0.6,reverse = True),inner ="points")
plt.show()
#Seaborn
dates = [ str(each).split("-")[0]  for each in data.release_date]
plt.figure(figsize = (20,10))
sns.countplot(dates)
plt.xticks(rotation= 90,fontsize = 12)
plt.xlabel("Movies release date",fontsize =17)
plt.ylabel("Number of movies",fontsize = 17)
plt.show()
dates[:5]
#Seaborn
sns.countplot(data.vote_level)
plt.show()
#Seaborn
plt.figure(figsize = (20,10))
sns.countplot(data.original_language)
plt.xticks(rotation = 45,fontsize = 12)
plt.show()
#Plotly
trace1= go.Histogram(
    x = dates,
    opacity= 0.80,
    marker = dict ( color = "yellowgreen")
)
data1 = [trace1]
fig =go.Figure( data = data1)
iplot(fig)
#Plotly
data1= [{
    "x":data.original_title,
    "y":data.vote_average.head(25),
    "mode":"markers",
    "marker": {
        "color":"rgb(200,155,120)",
        "size":data.popularity.head(25),
        "showscale":True
    },
    "text":data.original_title.head(25)
}
]
iplot(data1)
#Plotly
plt.subplots( figsize = (20,12))
wordcould = WordCloud(
    background_color= "black",
    width= 2160,
    height = 720,
).generate(",".join(c))
plt.imshow(wordcould)
plt.axis("off")
plt.show()
c[:5]
#Plotly
trace1=go.Scatter3d(
    x =data.original_title.head(100),
    y = data.vote_count.head(100),
    z= data.vote_average.head(100),
    mode = "markers",
    marker= dict(
        color= data.vote_average.head(100),
        colorscale = "Viridis",
        size = 10
    )
)
data1 = [trace1]
layout = go.Layout(
    margin = dict (
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = dict( data = data1,layout = layout)
iplot(fig)