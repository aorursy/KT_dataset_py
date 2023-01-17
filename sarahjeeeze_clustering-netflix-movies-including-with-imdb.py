# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install imdbpy

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly as pl

import plotly.graph_objs as gobj

import pandas as pd

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

import squarify  

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

import random



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import imdb

df = pd.read_csv('/kaggle/input/imdbfile/mycsvfile.csv')

df.head(10)
#create a dictionary of show_id and titles to be used later

showNames = pd.Series(df.title.values,index=df.show_id).to_dict()



def filmPredict(title):

    print(showNames[(title)])

    

filmPredict(81145628)

filmPredict(81035887)

filmPredict(80232095)

filmPredict(80108610)
import collections

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot

import pandas as pd 



col = "listed_in"

categories = ", ".join(df['listed_in']).split(", ")

counter_list = collections.Counter(categories).most_common(50)

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()





number_of_colors = len(labels)



colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

             for i in range(number_of_colors)]

#print(color)

fig = plt.figure(figsize=(30,15))



squarify.plot(sizes=values, label=labels, alpha=.8,color=colors )

plt.axis('off')

plt.show()
df.info()
#same with countries





col = "country"

df['country'] = df.country.fillna('none')

categories = ", ".join(df['country']).split(",")



counter_list = collections.Counter(categories).most_common(50)

counter_list = counter_list[1:50]

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Most common actors", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()



import plotly.express as px





#try to do one for other countriesp



labels



#make an interactive pie chart and countries with less than 20 make as 'other'



df1 = {'Country':labels,'number':values}

df1 = pd.DataFrame(df1)

df1.loc[df1['number'] < 50, 'Country'] = 'Other countries' 

fig = px.pie(df1, values='number', names='Country', color_discrete_sequence=px.colors.cyclical.Phase)

fig.show()





#initializing the data variable

data = dict(type = 'choropleth',

            

            locations = labels,

            locationmode = 'country names',

            colorscale= 'Portland',

            

            text= labels,

            z=values,

            colorbar = {'title':'Country Colours', 'len':200,'lenmode':'pixels' })



layout = dict(geo = {'scope':'world'}, title_text ='Netflix shows in each country')



col_map = gobj.Figure(data = [data],layout = layout)



iplot(col_map)


col = "cast"

df['cast'] = df.cast.fillna('bam')

categories = ", ".join(df['cast']).split(", ")



counter_list = collections.Counter(categories).most_common(50)

counter_list = counter_list[1:50]

labels = [_[0] for _ in counter_list][::-1]

values = [_[1] for _ in counter_list][::-1]

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Most common actors", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()

print(len(categories))
#create a function for putting in name of actor and getting films/shows they are in 



actorDictionary = {}

counter_list = collections.Counter(categories).most_common(1000)

counter_list = counter_list[1:1000]

Actors = [_[0] for _ in counter_list][::-1]

justActors = pd.DataFrame()

justActors['title'] = df['title']

justActors['cast'] = df['cast']







#cycle through the labels and if they - i know its the worst fofrmula ever

count = 0

for cols, rows in justActors.iterrows():

    for actor in Actors:

        if actor in (rows[1]):

            actorDictionary.setdefault(actor, []).append(rows[0])

            count+=1



actorDictionary

print(count)
def findFilms(name):

    x = actorDictionary[name]

    print(x)

    



findFilms('Ricky Gervais')

findFilms('Nicolas Cage')

findFilms('Brad Pitt')
import imdb



ia = imdb.IMDb()



movies = ia.search_movie('matrix')

movies[0]



for k,v in movies[0].items():

    print(k,v)
movie = movies[0]



ia.update(movie, info=['taglines','vote details'])



#create function for retriveing ratings



def rating(name):

    name = str(name)

    movies = ia.search_movie(name)

    movie = movies[0]

    ia.update(movie, info=['taglines','vote details'])

    rating = movie['arithmetic mean']

    return (rating)





def rating_test(name):

    try:

        x = rating(name)

        return(x)

    

    except:

         x = 0

         return(x)

        

x = rating_test('Babel')

print(x)
okk = df['title'][3]

rating_test(okk)
#only run once to get imdb ratings



#df['rating'] = df['title'].apply(rating_test)

#df.head(10)



#df.to_csv('mycsvfile.csv',index=False)

df = pd.read_csv('/kaggle/input/imdbfile/mycsvfile.csv')

df.head()




#df.rating.plot()

plt.hist(df['rating'])



#so most movies on netflix have a rating of 



x = df['rating'].median()

plt.axvline(x, color='k', linestyle='dashed', linewidth=1)

#plt.line(y=6.4)

plt.show()

len(actorDictionary)
ratingDic = {}



hay = df['title'].tolist()

hay = pd.Series(df.rating.values,index=df.title).to_dict()

    

hay



count = 0

#create a dictionary for average imdb score for each actor to find which actors are in the best films - do similiar for genre etc - then think of how to cluster the categrical data

actorMean = {}



for k,v in actorDictionary.items():

    count = 0

    for i in v:

        count += hay[i]

    actorMean[k] = (count/len(v))



    

actorMean

#sort by highest first 

sort = {k: v for k, v in sorted(actorMean.items(), key=lambda item: item[1])}

dfRate = pd.DataFrame.from_dict(sort.items())

#df.sort_values(by=[1])



title = ['actor','mean']

dfRate.columns = title

dfRate.info()



#df.sort_values(by = ['mean', 'actor'])

counter_list = dfRate.tail(50)

counter_list



labels = counter_list['actor']

values = counter_list['mean']

trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows", marker=dict(color="#a678de"))



data = [trace1]

layout = go.Layout(title="Actors with the highest average ratings", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

fig.show()

print(len(categories))
def findIMDBaverage(name):

    ave = actorMean[name]

    return(ave)



#example

findIMDBaverage('Keanu Reeves')
#do the same for genre? or try to create a model/cluster to find other films someone might like based on what they liked previously. 



dfCopy = df

#only keep columns you want for algorith

dfCopy.head()

dfCopy['director'] = dfCopy.director.fillna('MR')



dfCopy['director3'], dfCopy['director2'] = dfCopy['director'].str.split(',',1).str

dfCopy.head()



dfCopy["director3"] = dfCopy["director3"].astype('category')

dfCopy["directorCode2"] = dfCopy["director3"].cat.codes

dfCopy = dfCopy.replace(-1,1)

dfCopy.head()

dfCopy["listed_in"] = dfCopy["listed_in"].astype('category')

dfCopy["genreCombo"] = dfCopy["listed_in"].cat.codes

dfCopy.head()

#dfCopy["genreCombo"].value_counts()



dfCopy["country"] = dfCopy["country"].astype('category')

dfCopy["country1"] = dfCopy["country"].cat.codes

dfCopy = dfCopy.set_index('show_id')

scalerValues = dfCopy.drop('type', axis=1) 

scalerValues = scalerValues.drop('title', axis=1) 

scalerValues = scalerValues.drop('director', axis=1) 

scalerValues = scalerValues.drop('cast', axis=1) 

scalerValues = scalerValues.drop('date_added', axis=1) 

scalerValues = scalerValues.drop('duration', axis=1) 

scalerValues = scalerValues.drop('listed_in', axis=1) 

scalerValues = scalerValues.drop('description', axis=1) 

scalerValues = scalerValues.drop('director2', axis=1) 

scalerValues = scalerValues.drop('director3', axis=1) 

scalerValues = scalerValues.drop('country', axis=1) 



scalerValues.head()









import numpy as np 

import pandas as pd 

from sklearn import preprocessing



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



col_names = list(scalerValues.columns)





mm_scaler = preprocessing.MinMaxScaler()

df_mm = mm_scaler.fit_transform(scalerValues)



scalerValues = pd.DataFrame(df_mm, columns=col_names)



fig, (ax1) = plt.subplots(ncols=1, figsize=(10, 8))

ax1.set_title('After MinMaxScaler')



sns.kdeplot(scalerValues['rating'], ax=ax1)



sns.kdeplot(scalerValues['genreCombo'], ax=ax1)

sns.kdeplot(scalerValues['directorCode2'], ax=ax1)

sns.kdeplot(scalerValues['release_year'], ax=ax1)

scalerValues.info()
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

X = scalerValues



pca = PCA()

pca.fit(X)

print(pca.explained_variance_ratio_)



#pca.info()

plt.plot(range(1,6),pca.explained_variance_ratio_.cumsum(),marker='o')

plt.title("explained variance by components")

plt.xlabel('Number of Components')

plt.ylabel('Cumulative explained variance')



scalerValues.head()

#we keep 3 pc's beause this give 80% of the variance



pca = PCA(n_components=3)

pca.fit(X)

scores_PCA = pca.transform(X)



#sum of squares

#we potentially want a lot of clusters so im going to say up to 100

sos = []

for i in range(1,100):

    kmeans_pca = KMeans(n_clusters = i,init ='k-means++',random_state=200)

    kmeans_pca.fit(scores_PCA)

    sos.append(kmeans_pca.inertia_)

    

    

plt.figure(figsize=(20,20))

plt.plot(range(1,100),sos,marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('SOS')

plt.title('k-meas with PCA clusterinn')
#gonna go for 20 because i know i want a lot of clusters 



kmeans_pca = KMeans(n_clusters=10,init='k-means++',random_state=42)

kmeans_pca.fit(scores_PCA)



Z = kmeans_pca.predict(scores_PCA)
#next task create a word frequency graph with description column



df_kmeans = pd.concat([X,pd.DataFrame(scores_PCA)],axis=1)

df_kmeans.columns.values[-3:]=['comp 1','comp 2','comp 3']

df_kmeans['segment k means PCA'] = kmeans_pca.labels_



df_kmeans = df_kmeans.set_index(df['show_id'])

df_kmeans.head()



x_axis = df_kmeans['comp 2']

y_axis = df_kmeans['comp 1']

plt.figure(figsize=(20,20))

sns.scatterplot(x_axis,y_axis,hue=df_kmeans['segment k means PCA'])

plt.title('clusters by pca components')

plt.show()





df_kmeans.info()



!pip install chart_studio

!pip install plotly

import glob

import numpy as np

import pandas as pd

import chart_studio

import plotly

#import plotly.plotly as py

#import chart_studio.plotly as py

import plotly.graph_objs as pgo

import chart_studio.plotly as py

#import chart_studio.plotly as py





chart_studio.tools.set_credentials_file(username='sarahjeeeze', api_key='SvTKCDpH5TQ7aCJuROxR')



trace0 = pgo.Scatter(x=df_kmeans['comp 2'],

                    y=df_kmeans['comp 1'],

                    text=df_kmeans.index,

                    mode='markers',

                    # Size by total population of each neighborhood. 

                    marker=plotly.graph_objs.scatter.Marker(size=df_kmeans['rating'],

                                      sizemode='diameter',

                                      sizeref=df_kmeans['rating'].max()/5,

                                      opacity=0.5,

                                     color=Z

                                                       ))

model = kmeans_pca

n_cluster = 10



# Represent cluster centers.

trace1 = pgo.Scatter(x=model.cluster_centers_[:, 0],

                     y=model.cluster_centers_[:, 1],

                     name='',

                     mode='markers',

                     marker=pgo.Marker(symbol='x',

                                       size=12,

                                      ),

                     

                     showlegend=False

)



layout5 = pgo.Layout(title='Baltimore Vital Signs (PCA)',

                     xaxis=pgo.XAxis(showgrid=False,

                                     zeroline=False,

                                     showticklabels=False),

                     yaxis=pgo.YAxis(showgrid=False,

                                     zeroline=False,

                                     showticklabels=False),

                     hovermode='closest'

)

datad = pgo.Data([trace0, trace1])

layout7 = layout5

layout7['title'] = 'Netflix  in 10 clustsers)'

fig7 = pgo.Figure(data=datad, layout=layout7)

py.iplot(fig7, filename='baltimore-cluster-map')



#defo needs some work 

#could not use titles as indexes because then you have to pay for plotly. #want to try hierachical next and create a function that out puts the 5 closest for the name of a film.

filmPredict(60004083)

filmPredict(80119188)



#could add more things to include in pca

#look here for idea of how to create labels for the numbers using sci kit learn as well as one hot encoding which could be used for genres and or feature hashing

#https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63