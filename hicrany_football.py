

import numpy as np 

import pandas as pd 



import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from wordcloud import WordCloud



import matplotlib.pyplot as plt

import seaborn as sns 



import os

print(os.listdir("../input"))

import warnings            

warnings.filterwarnings("ignore") # if there is a warning after some codes, this will avoid us to see them.

plt.style.use('ggplot')



data = pd.read_csv('../input/data.csv')
data.info()
data.dtypes
data.corr()
f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.Age.plot(kind = 'line', color = 'r',label = 'Age',linewidth=1,alpha = 0.7,grid = True,linestyle = ':')

data.Curve.plot(color = 'b',label = 'Curve',linewidth=.5, alpha = 0.5,grid = True,linestyle = '--')

plt.legend(loc='upper right')     

plt.xlabel('x ax')          

plt.ylabel('y ax')

plt.title('Line Plot')      

plt.show()
data.plot(kind='scatter', x='Age', y='Curve',alpha = 0.5,color = 'red')

plt.xlabel('Age')              # label = name of label

plt.ylabel('Curve')

plt.title('Age Curve Scatter Plot') 
data.Age.plot(kind = 'hist',bins = 60,figsize = (8,8))

plt.show()
data.Overall.plot(kind = 'hist',bins = 60,figsize = (8,8))

plt.show()
data.plot(kind="hist", subplots=True, grid=True, alpha=0.5,figsize=(40,40))

plt.show()
data.Curve.plot(kind = 'hist',bins = 40,figsize = (9,9))

plt.show()
x = data['Age']<20

data[x]  
data[np.logical_and(data['Age']<20, data['Curve']<10 )]
data[(data['Age']<20) & (data['Curve']<9)]
threshold = sum(data.Age)/len(data.Age)

print(threshold)

data["age_level"] = ["old" if i > threshold else "young" for i in data.Age]

data.loc[:20,["age_level","Age"]]
data.describe()
data.boxplot(column='Age')
data.head(5)
data.tail(5)
data.info()
data1= data

data1["CM"].dropna(inplace= True)
assert 1 == 1
assert data['CM'].notnull().all()
data["CM"].fillna('empty', inplace = True)
assert data['CM'].notnull().all()
time_list = ["1996-09-24","1992-09-23"]

print(type(time_list[1]))

datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data['Overall'].unique()
data['Nationality'].unique()
data['Overall'].unique()
data.Nationality
data.Age
data.dtypes
data.columns
data.Overall.replace(['-'],0.1,inplace = True)

data.Overall= data.Overall.astype(float)

area_list = list(data['Overall'].unique())

area_age_ratio = []

for i in area_list:

    x = data[data['Overall']==i]

    area_Age= sum(x.Age)/len(x)

    area_age_ratio.append(area_Age)

data1 = pd.DataFrame({'area_list': area_list,'area_age_ratio':area_age_ratio})

new_index = (data1['area_age_ratio'].sort_values(ascending=False)).index.values

sorted_data = data1.reindex(new_index)



plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_age_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Overall')

plt.ylabel('Age Rate')

plt.title('Age Rate Given Overall')
data.dtypes
data.columns
data1.dtypes
data1.columns
data.info()
#sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

#sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

#data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

#data.sort_values('area_poverty_ratio',inplace=True)



# visualize

#f,ax1 = plt.subplots(figsize =(20,10))

#sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='lime',alpha=0.8)

#sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)

#plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

#plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

#plt.xlabel('States',fontsize = 15,color='blue')

#plt.ylabel('Values',fontsize = 15,color='blue')

#plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

#plt.grid()

g = sns.jointplot(data.Age, data.Curve, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
data.head()
g = sns.jointplot("Age", "Potential", data=data,size=10, ratio=5, color="b")
g = sns.jointplot("Age", "GKPositioning", data=data,size=10, ratio=5, color="b")
data.Nationality.count()
data.Nationality.head(15)
data.Nationality.value_counts()
#pie plot

data.Nationality.dropna(inplace = True)

labels = data.Nationality.value_counts().index

colors = ['grey','blue','red','yellow','green','brown','purple','pink','white','orange']

explode = [0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

sizes = data.Nationality.value_counts().values



plt.figure(figsize = (15,15))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title(' Nationality of People',color = 'blue',fontsize = 20)
data.head()
sns.lmplot(x="Age", y="Potential", data=data)

plt.show()
sns.lmplot(x="Age", y="Curve", data=data)

plt.show()
data1.head()
sns.lmplot(x="area_list", y="area_age_ratio", data=data1)

plt.show()
data1.head()
sns.kdeplot(data1.area_list, data1.area_age_ratio, shade=True, cut=5)

plt.show()
data.head()
data1.head()
pal = sns.cubehelix_palette(200, rot=-.4, dark=.3)

sns.violinplot(data=data1, palette=pal, inner="points")

plt.show()
data1.info()
data1.corr()
#corr map 

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data1.corr(), annot=True, linewidths=0.5,linecolor="pink", fmt= '.1f',ax=ax)

plt.show()
data.head()
data.age_level.unique()
data1.head()
sns.pairplot(data1)

plt.show()
data.head(10)
sns.countplot(data.age_level)

plt.title("age_level",color = 'blue',fontsize=15)
data.head()
df = data.iloc[:100,:]



# import graph objects as "go"

import plotly.graph_objs as go



# Creating trace1

trace1 = go.Scatter(

                    x = df.Age,

                    y = df.Potential,

                    mode = "lines",

                    name = "Potential",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= df.Name)

# Creating trace2

trace2 = go.Scatter(

                    x = df.Age,

                    y = df.Overall,

                    mode = "lines+markers",

                    name = "Overall",

                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'),

                    text= df.Name)

data5 = [trace1, trace2]

layout = dict(title = 'Age_Potential vs Age_Overall 0f Top 100 Players',

              xaxis= dict(title= 'Age',ticklen= 3,zeroline= False)

             )

fig = dict(data = data5, layout = layout)

iplot(fig)
df2 = data.iloc[:100,:]

import plotly.graph_objs as go



trace1 =go.Scatter(

                    x = df2.Age,

                    y = df2.Potential,

                    mode = "markers",

                    name = "Potential",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= df2.Name)

# creating trace2

trace2 =go.Scatter(

                    x = df2.Age,

                    y = df2.Overall,

                    mode = "markers",

                    name = "Overall",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= df2.Name)



data1 = [trace1, trace2]

layout = dict(title = ' Overall vs Potential ',

              xaxis= dict(title= 'Overall',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Potential',ticklen= 5,zeroline= False)

             )

fig = dict(data = data1, layout = layout)

iplot(fig)
df2
df2 = data.iloc[:4,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2.Age,

                y = df2.Potential,

                name = "Potential",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=2.5)),

                text = df2.Name)

# create trace2 

trace2 = go.Bar(

                x = df2.Age,

                y = df2.Overall,

                name = "Overall",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=2.5)),

                text = df2.Name)

data6 = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data6, layout = layout)

iplot(fig)
df2 = data.iloc[:5,:]

# import graph objects as "go"

import plotly.graph_objs as go

# create trace1 

trace1 = go.Bar(

                x = df2.Age,

                y = df2.Potential,

                name = "Potential",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=2.5)),

                text = df2.Name)

# create trace2 

trace2 = go.Bar(

                x = df2.Age,

                y = df2.Overall,

                name = "Overall",

                marker = dict(color = 'rgba(255, 255, 128, 0.5)',

                              line=dict(color='rgb(0,0,0)',width=2.5)),

                text = df2.Name)

data6 = [trace1, trace2]

layout = go.Layout(barmode = "group")

fig = go.Figure(data = data6, layout = layout)

iplot(fig)
df2 = data.iloc[:3,:]

import plotly.graph_objs as go



x = df2.Name



trace1 = {

  'x': x,

  'y': df2.Potential,

  'name': 'Potential',

  'type': 'bar'

};

trace2 = {

  'x': x,

  'y': df2.Overall,

  'name': 'Overall',

  'type': 'bar'

};

data7 = [trace1, trace2];

layout = {

  'xaxis': {'title': 'Top 3 players'},

  'barmode': 'relative',

  'title': ' top 3 playerss'

};

fig = go.Figure(data = data7, layout = layout)

iplot(fig)
df2= data.iloc[:7,:]

pie1_list = df2.Age

labels = df2.Name

# figure

fig = {

  "data": [

    {

      "values": pie1_list,

      "labels": labels,

      "domain": {"x": [0, .5]},

      "name": "Age Of Players Rates",

      "hoverinfo":"label+percent+name",

      "hole": .3,

      "type": "pie"

    },],

  "layout": {

        "title":"Players rates",

        "annotations": [

            { "font": { "size": 20},

              "showarrow": False,

              "text": "Age of Players",

                "x": 0.20,

                "y": 1

            },

        ]

    }

}

iplot(fig)
df2
df2.head()
x2011 = df2.Overall

x2012 = df2.Age



trace1 = go.Histogram(

    x=x2011,

    opacity=0.75,

    name = "Overall",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))

trace2 = go.Histogram(

    x=x2012,

    opacity=0.75,

    name = "Age",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data11 = [trace1, trace2]

layout = go.Layout(barmode='overlay',

                   title=' students-staff ratio in 2011 and 2012',

                   xaxis=dict(title='students-staff ratio'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data11, layout=layout)

iplot(fig)
data.head()
word= data.Nationality

plt.subplots(figsize=(8,8))

wordcloud = WordCloud(

                          background_color='white',

                          width=512,

                          height=384

                         ).generate(" ".join(word))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('graph.png')



plt.show()
x2015 = data



trace0 = go.Box(

    y=x2015.Age,

    name = 'Age of players',

    marker = dict(

        color = 'rgb(12, 12, 140)',

    )

)

trace1 = go.Box(

    y=x2015.Overall,

    name = 'Overall of players',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

datax = [trace0, trace1]

iplot(datax)
import plotly.figure_factory as ff

# prepare data

dataframe = data

data2018 = dataframe.loc[:,["Age","Marking","Potential"]]

data2018["index"] = np.arange(1,len(data2018)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2018, diag='box', index='index',colormap='Earth',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
data2018
import plotly.figure_factory as ff

# prepare data

dataframe = data

data2015 = dataframe.loc[:,["Age","Overall","Potential"]]

data2015["index"] = np.arange(1,len(data2015)+1)

# scatter matrix

fig = ff.create_scatterplotmatrix(data2015, diag='box', index='index',colormap='Portland',

                                  colormap_type='cat',

                                  height=700, width=700)

iplot(fig)
data2015
trace1 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.Overall,

    name = "Overall",

    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

)

# second line plot

trace2 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.Potential,

    xaxis='x2',

    yaxis='y2',

    name = "Potential",

    marker = dict(color = 'rgba(160, 112, 20, 0.8)'),

)

datar = [trace1, trace2]

layout = go.Layout(

    xaxis2=dict(

        domain=[0.6, 0.95],

        anchor='y2',        

    ),

    yaxis2=dict(

        domain=[0.6, 0.95],

        anchor='x2',

    ),

    title = 'Potential and Overall vs  Age of Players '



)



fig = go.Figure(data=datar, layout=layout)

iplot(fig)

trace1 = go.Scatter3d(

    x=dataframe.Age,

    y=dataframe.Overall,

    z=dataframe.Potential,

    mode='markers',

    marker=dict(

        size=10,

        color='rgb(255,0,0)',                # set color to an array/list of desired values      

    )

)



datay= [trace1]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0  

    )

    

)

fig = go.Figure(data=datay, layout=layout)

iplot(fig)
data.head()
trace1 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.Overall,

    name = "Overall"

)

trace2 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.Potential,

    xaxis='x2',

    yaxis='y2',

    name = "Potential"

)

trace3 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.Marking,

    xaxis='x3',

    yaxis='y3',

    name = "Marking"

)

trace4 = go.Scatter(

    x=dataframe.Age,

    y=dataframe.GKKicking,

    xaxis='x4',

    yaxis='y4',

    name = "GKKicking"

)

datal = [trace1, trace2, trace3, trace4]

layout = go.Layout(

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 0.45]

    ),

    xaxis2=dict(

        domain=[0.55, 1]

    ),

    xaxis3=dict(

        domain=[0, 0.45],

        anchor='y3'

    ),

    xaxis4=dict(

        domain=[0.55, 1],

        anchor='y4'

    ),

    yaxis2=dict(

        domain=[0, 0.45],

        anchor='x2'

    ),

    yaxis3=dict(

        domain=[0.55, 1]

    ),

    yaxis4=dict(

        domain=[0.55, 1],

        anchor='x4'

    ),

    title = 'Overall, Potential, Marking and total score VS World Rank of Universities'

)

fig = go.Figure(data=datal, layout=layout)

iplot(fig)
data.head()
dictionary = {"column1":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],

              "column2":[1,2,3,4,np.nan,6,7,8,np.nan,10,np.nan,12,13,14,15,16,np.nan,18,np.nan,20],

              "column3":[1,2,3,4,np.nan,6,7,8,9,10,11,12,13,np.nan,15,16,17,18,np.nan,20]}

# Create data frame from dictionary

data_missingno = pd.DataFrame(dictionary) 



# import missingno library

import missingno as msno

msno.matrix(data_missingno)

plt.show()
data_missingno.head(20)
msno.bar(data_missingno)

plt.show()
corr = data.iloc[:,0:4].corr()

corr
links
import networkx as nx



# Transform it in a links data frame (3 columns only):

links = corr.stack().reset_index()

links.columns = ['var1', 'var2','value']



# correlation

threshold = -1



# Keep only correlation over a threshold and remove self correlation (cor(A,A)=1)

links_filtered=links.loc[ (links['value'] >= threshold ) & (links['var1'] != links['var2']) ]

 

# Build your graph

G=nx.from_pandas_edgelist(links_filtered, 'var1', 'var2')

 

# Plot the network

nx.draw_circular(G, with_labels=True, node_color='orange', node_size=300, edge_color='red', linewidths=1, font_size=10)
data.head()
from matplotlib_venn import venn2

Age = data.iloc[:,0]

ID = data.iloc[:,1]

Overall = data.iloc[:,2]

Potential = data.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(Age)-1325, len(ID)-678, 5615), set_labels = ('Age', 'Overall'))

plt.show()
# donut plot

feature_names = "Age","ID","Overall","Potential"

feature_size = [len(Age),len(ID),len(Overall),len(Potential)]

# create a circle for the center of plot

circle = plt.Circle((0,0),0.2,color = "white")

plt.pie(feature_size, labels = feature_names, colors = ["red","green","black","cyan"] )

p = plt.gcf()

p.gca().add_artist(circle)

plt.title("Number of Each Features")

plt.show()