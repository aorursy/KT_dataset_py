# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



#json

import json



# word cloud library

from wordcloud import WordCloud



# seabron

import seaborn as sns



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_movies = pd.read_csv('../input/tmdb_5000_movies.csv')

data_credits = pd.read_csv('../input/tmdb_5000_credits.csv')
data_movies.info()
data_credits.info()
data_movies_average = data_movies .copy()

data_movies_average = data_movies_average.sort_values(['vote_average'],ascending = False)
labelx = data_movies_average['original_title'].head(50)

labely = data_movies_average['vote_average'].head(50)



labelx = np.array(labelx)

labely = np.array(labely)



plt.figure(figsize= (18,12))

sns.barplot(x = labelx ,y = labely)

plt.xticks(rotation = 90)

plt.ylabel('Vote Average')

plt.xlabel('Title')
C = data_movies.vote_average.mean()

print('the votes averages: ' ,C)
m = data_movies.vote_count.quantile(.93)

print('the requred vote is : ',m)
data_most_popular = data_movies.copy()

data_most_popular = data_most_popular.loc[data_most_popular['vote_count']>=m]
V = data_most_popular.vote_count

R = data_most_popular.vote_average
result = (V/(V+m)*R) + (m / (V+m)* C)
data_most_popular['pop'] = result
data_most_popular
data_most_popular = data_most_popular.sort_values('pop',ascending = False)
data_most_popular[['title','vote_average']].head(10)


data = [go.Bar(

        x=data_most_popular.title.head(50),

        y= data_most_popular['vote_average'].head(50),

        marker = dict(color = 'rgba(110, 0,255, 0.6)',

           line=dict(color='rgb(0,0,0)',width=1.5)),

            text = data_most_popular.title.head(50)

    )]



iplot(data, filename='basic-bar')
data_movies = pd.read_csv('../input/tmdb_5000_movies.csv')

data_credits = pd.read_csv('../input/tmdb_5000_credits.csv')

data_movies['genres']=data_movies['genres'].apply(json.loads)

for index,i   in zip(data_movies.index,data_movies['genres']):

    list1=[]    

    for j in range(len(i)):       

        list1.append((i[j]['name']))

    data_movies.loc[index,'genres'] = str(list1) 

    

data_movies['keywords']=data_movies['keywords'].apply(json.loads)

for index,i in zip(data_movies.index,data_movies['keywords']):

    list1=[]  

    for j in range(len(i)):

        list1.append((i[j]['name']))

    data_movies.loc[index,'keywords'] = str(list1) 

    

data_movies['production_companies'] = data_movies['production_companies'].apply(json.loads)

for index,i in zip(data_movies.index,data_movies['production_companies']):

    list1=[]

    for j in range(len(i)):

         list1.append((i[j]['name']))

    data_movies.loc[index,'production_companies'] = str(list1)     

    

data_movies['production_countries'] = data_movies['production_countries'].apply(json.loads)

for index,i in zip(data_movies.index,data_movies['production_countries']):

    list1=[]

    for j in range(len(i)):

        list1.append((i[j]['name']))

    data_movies.loc[index,'production_countries'] = str(list1)    

    

data_credits['cast'] = data_credits['cast'].apply(json.loads)

for index,i in zip(data_credits.index,data_credits['cast']):

    list1=[]

    for j in range(len(i)):

        list1.append((i[j]['name']))

    data_credits.loc[index,'cast'] = str(list1)     

    
data_movies['genres'] = data_movies['genres'].str.strip('[]').str.replace(' ','').str.replace("''",'')

data_movies['genres'] = data_movies['genres'].str.split(',')
list1 = []

for i in data_movies['genres']:

    list1.extend(i)
most_genres = pd.Series(list1).value_counts().sort_values(ascending = False )

genres = most_genres.index

values = most_genres.values
f,ax = plt.subplots(figsize = (12,12))

sns.barplot(x = values,y = genres,color = 'blue', alpha = .6)
data_movies.release_date = data_movies.release_date.apply(lambda x: x.replace('-',' ') if '-' in str(x) else x)

data_movies_release = data_movies.copy()

data_movies_release.release_date = data_movies_release.release_date.apply(lambda x: x[0:4] if len(str(x))>4 else str(x))

data_movies_release.release_date = data_movies_release.release_date.apply(lambda x: x.strip('nan') if 'nan' in str(x) else x)    

data_movies_release.release_date = pd.to_numeric(data_movies_release.release_date)  

data_inf = data_movies_release.release_date.value_counts()
labelx = np.array(data_inf.index)

labely = np.array(data_inf.values)
plt.subplots(figsize = (18,15))

sns.pointplot(x=labelx,y=labely,color='blue',alpha = 0.1)

plt.xticks(rotation = 90)

plt.title('movies for years',fontsize = 20,color='red')

plt.xlabel('YEARS',fontsize = 20,color='green')

plt.ylabel('MOVİES',fontsize = 20,color='green')

plt.grid()
g=sns.jointplot(labelx,labely,kind='kde',siz=20)

plt.savefig('graph.png')

plt.show()

g = sns.jointplot(labelx,labely,size=7,color= 'blue',ratio = 3)

plt.show()
genre_list = []

for index,row in data_movies.iterrows():

    genres = row['genres']

    for i in genres:

        genre_list.append(i)



  

    

   

index = pd.Series(genre_list).value_counts().index

value = pd.Series(genre_list).value_counts().values
list1=[]

for j in range(len(value)):

    list1.append(0)

   
plt.subplots(figsize = (7,7))

colors = ['grey','blue','red','yellow','green','brown']

plt.pie(value,explode = list1,labels=index,autopct='%1.1f%%')

plt.show()
df1=data_movies.copy()
df1 = df1.sort_values('budget',ascending = False)



df1[['budget','genres','title','revenue']].head(10)
list_budget = df1.budget.head(10)

list_revanue = df1.revenue.head(10)
import plotly.graph_objs as go



trace1 = go.Bar (

                x = df1.title.head(10),

                y = list_budget,

                name = "Budget",

                marker = dict(color= 'rgba(0,0,200,0.4)',

                              line = dict(color = 'rgb(0,0,0)',width = 1)),

                text = df1.genres

              

               )





trace2 = go.Bar(

                x = df1.title.head(10),

                y = list_revanue,

                name = "revanue",

                  marker = dict(color= 'rgba(200,100,10,0.7)',

                                line= dict(color = 'rgb(0,0,0)',width = 1)),

                   text = df1.genres

)



data = [trace1, trace2]

layout = go.Layout(barmode='group')



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='grouped-bar')

    
data_movies['production_countries'] = data_movies['production_countries'].str.strip('[]').str.replace(' ','').str.replace("''",'')

data_movies['production_countries'] = data_movies['production_countries'].str.split(',')
list1 = []

for i in data_movies['production_countries']:

    list1.extend(i)

   

df2 = pd.Series(list1).value_counts().dropna().head(10)

countries = df2.index

count = df2.values
fig = {

    "data" : [

        {

            "values" : count,

            "labels" : countries,

            "domain" :{"x":[0,0.5]},

            "name" : "rate of product of countries",

            "hoverinfo": "label+percent+name",

            "hole": .3,

            "type": "pie"

            

            

        } ,],

    "layout":  {

        "title": "World rate of product of countries ",

        "annotations" : [

            {

                "font": {"size":20},

                "showarrow": False,

                "text": "rate of countries",

                "x": 0.20,

                "y": 1

            },

        ]

    }  

}

iplot(fig)