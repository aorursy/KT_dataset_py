import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

plt.style.use("fivethirtyeight")

sns.set_context("talk")

%matplotlib inline
data = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data.csv")

genre_data = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv")
data.head()
genre_data.head()
genre_data.shape
data.shape
data['duration_min'] = data['duration_ms']/60000

data['duration_min'] = data['duration_min'].round(2)

genre_data['duration_min'] = data['duration_ms']/60000

genre_data['duration_min'] = data['duration_min'].round(2)
data.drop(['id','release_date','duration_ms'],inplace=True,axis=1)
data.isnull().sum().sum()
genre_data.isnull().sum().sum()
plt.figure(figsize=(15, 8))

sns.set(style="whitegrid")

corr = data.corr()

sns.heatmap(corr,annot=True,cmap="YlGnBu")
rel = data[data['year'] >= 2015]

variables = ['acousticness', 'danceability', 'energy','instrumentalness', 'key', 'liveness',

             'loudness', 'speechiness', 'tempo', 'valence']

year = range(2015,2021)



fig = plt.figure(figsize=(15,25))

for variable,num in zip(variables, range(1,len(variables)+1)):

    ax = fig.add_subplot(5,2,num)

    sns.scatterplot(variable, 'popularity', data=rel)

    plt.title('Relation between {} and Popularity'.format(variable))

    plt.xlabel(variable)

    plt.ylabel('Popularity')

fig.tight_layout(pad=0.5)
top = data[['name','duration_min']].sort_values('duration_min').tail(10)

fig = plt.figure(figsize=(15,7))

plt.bar( top['name'],

         top['duration_min'],

         width=0.45,

         color = ['#e01e37','#da1e37','#c71f37','#bd1f36','#b21e35','#a71e34','#a11d33','#85182a','#6e1423','#641220'])

plt.xticks(rotation=45,ha='right')

plt.title('Top 10 songs with highest duration',y=1.1,fontsize=20)

plt.xlabel('Songs')

plt.ylabel('Duration (minutes)')
!jupyter nbextension enable --py --sys-prefix widgetsnbextension
import ipywidgets as widgets

from ipywidgets import HBox, VBox

from IPython.display import display
@widgets.interact_manual(

    Year = range(1921,2021))

def plot(Year = 2020):

    pop = data[data['year'] == Year]

    pop = pop[['name','popularity','year']].sort_values('popularity').tail(10)

    fig = plt.figure(figsize=(15,5))

    plt.bar( pop['name'],

             pop['popularity'],

             width=0.4,

             color = ['#ffb600','#ffaa00','#ff9e00','#ff9100','#ff8500','#ff7900','#ff6d00','#ff6000','#ff5400','#ff4800'])

    plt.xticks(rotation=45,ha='right')

    t = 'Top 10 Most Popular Songs of ' + str(Year)

    plt.title(t,y=1.1,fontsize=20)

    plt.xlabel('Songs')

    plt.ylabel('Popularity (Ranges from 0 to 100)')

    ax.axes.get_xaxis().set_visible(True)
pop = data[['name','popularity']].sort_values('popularity').tail(10)

fig = plt.figure(figsize=(15,5))

plt.bar( pop['name'],

         pop['popularity'],

         width=0.45,

         color = ['#ffb600','#ffaa00','#ff9e00','#ff9100','#ff8500','#ff7900','#ff6d00','#ff6000','#ff5400','#ff4800'])

plt.xticks(rotation=45,ha='right')

plt.title('Top 10 Most Popular Songs from 1921-2020',y=1.1,fontsize=20)

plt.xlabel('Songs')

plt.ylabel('Popularity (Ranges from 0 to 100)')

ax.axes.get_xaxis().set_visible(True)
@widgets.interact_manual(

    Year = range(1921,2021))

def plot(Year = 2020):

    arpop = data[data['year'] == Year]

    arpop = pd.DataFrame(arpop.groupby('artists')['popularity'].sum()).sort_values('popularity').tail(10).reset_index()

    fig = plt.figure(figsize=(15,5))

    plt.bar( arpop['artists'],

             arpop['popularity'],

             width=0.45,

             color = ['#caf0f8','#ade8f4','#90e0ef','#48cae4','#00b4d8','#0096c7','#0077b6','#023e8a','#03045e','#14213d'])

    plt.xticks(rotation=45,ha='right')

    t='Top 10 Most Popular Artists from '+str(Year)

    plt.title(t,y=1.1,fontsize=20)

    plt.xlabel('Artists')

    plt.ylabel('Popularity (Ranges from 0 to 100)')

    ax.axes.get_xaxis().set_visible(True)
arpop = pd.DataFrame(data.groupby('artists')['popularity'].sum()).sort_values('popularity').tail(10).reset_index()

fig = plt.figure(figsize=(15,5))

plt.bar( arpop['artists'],

         arpop['popularity'],

         width=0.45,

         color = ['#caf0f8','#ade8f4','#90e0ef','#48cae4','#00b4d8','#0096c7','#0077b6','#023e8a','#03045e','#14213d'])

plt.xticks(rotation=45,ha='right')

plt.title('Top 10 Most Popular Artists from 1921-2020',y=1.1,fontsize=20)

plt.xlabel('Artists')

plt.ylabel('Popularity (Ranges from 0 to 100)')

ax.axes.get_xaxis().set_visible(True)
@widgets.interact_manual(

    Category = ['acousticness','danceability','energy','instrumentalness','liveness','speechiness','tempo'])

def plot(Category = 'acousticness'):

    df = pd.DataFrame(data[[Category,'name']]).sort_values(Category).tail().set_index('name')

    

    ax = df.plot(kind='barh', 

          figsize = (8, 5), 

          width = 0.5,

          color='#75daad')

    t='Top 5 Songs According to ' + Category

    plt.title(t,y=1.1,fontsize=20)

    plt.xlabel(Category)

    plt.ylabel('Songs')

    plt.xlim(0.96,1)

    plt.yticks(fontsize=20)

    ax.get_legend().remove()
df = pd.DataFrame(data[['valence','name']]).sort_values('valence').tail().set_index('name')

ax = df.plot(kind='barh', 

          figsize = (12, 5), 

          width = 0.5,

          color='#fddb3a')

t='Top 5 Happiest Songs'

plt.title(t,y=1.1,fontsize=20)

plt.xlabel('Valance')

plt.ylabel('Songs')

plt.yticks(fontsize=20)

plt.xlim(0.9975,1)

ax.get_legend().remove()
df = pd.DataFrame(data[['valence','name','duration_min']]).sort_values('valence').set_index('name')

df = df[df['duration_min']>3].head()

df
x=pd.DataFrame(data['artists'].value_counts().head()).reset_index()

x.columns=['Artists','Song_Count']

x
x=pd.DataFrame(data.groupby('artists')['duration_min'].sum())

x.sort_values('duration_min').tail().reset_index()
colors_list = ['#fdc500','#00509d']

mpl.rcParams['font.size'] = 18.0

ex = pd.DataFrame(data['explicit'].value_counts())

ex['explicit'].plot(kind='pie',

            figsize=(15, 6),

            autopct='%1.1f%%', 

            startangle=90,  

            shadow=True,       

            labels=None,    

            pctdistance=1.2, 

            colors=colors_list,

            )



plt.title('Explicit Content Ratio',y=1.1,fontsize=25) 

plt.axis('equal') 

plt.legend(labels=['No explicit content','Explicit content'], loc='upper left') 

plt.show()
year = pd.DataFrame(data['year'].value_counts())

year = year.sort_index()

ax=year.plot(kind='line',figsize=(15,8) ,color='#6f4a8e', linewidth=2)

plt.title("Number of songs released Yearwise",y=1.05,fontsize=20)

plt.xlabel('Years')

plt.ylabel('Count')

ax.axes.get_xaxis().set_visible(True)
variables = ['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness']

color = ['#9B5DE5','#F15BB5','#F8A07B','#FEE440','#7FD09D','#00BBF9','#17F6D8']

df= data.groupby('year')[variables].mean().reset_index()

import matplotlib.patches as mpatches

fig = plt.figure(figsize=(15,8))

l=[]

for i in range(len(variables)):

    x = str(i)

    x = mpatches.Patch(color=color[i], label=variables[i])

    l.append(x)

    plt.plot( 'year',variables[i],data=df,marker='',color=color[i],linewidth=1.3)

plt.legend(handles=l,bbox_to_anchor=(1,1.2))

plt.title("Trend Analysis Year Wise",y=1.1,fontsize=25)

plt.xlabel('Years')
df= data.groupby('year')['tempo'].mean().reset_index()

fig = plt.figure(figsize=(15,4))

plt.plot( 'year','tempo',data=df,marker='',color='#12947f',linewidth=1.3)

plt.title("Trend Analysis Year Wise - Tempo ",y=1.1,fontsize=25)

plt.xlabel('Years')
df= data.groupby('year')['loudness'].mean().reset_index()

fig = plt.figure(figsize=(15,4))

plt.plot( 'year','loudness',data=df,marker='',color='#00005c',linewidth=1.3)

plt.title("Trend Analysis Year Wise - Loudness ",y=1.1,fontsize=25)

plt.xlabel('Years')

plt.ylabel('Loudness (db)')
df=data[['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness']]

df['popularity']=pd.DataFrame(data['popularity']/10).apply(np.floor).astype(int)

df['popularity']=df['popularity']*10

df=df.groupby('popularity')['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness'].mean().reset_index()

color1=['#440047','#e11d74','#96bb7c','#fddb3a','#00bcd4','#ff5722','#ffa5b0']

fig = plt.figure(figsize=(15,8))

l=[]

variables = ['acousticness','danceability','energy','instrumentalness','valence','liveness','speechiness']

for i in range(len(variables)):

    x = str(i)

    x = mpatches.Patch(color=color1[i], label=variables[i])

    l.append(x)

    plt.plot( 'popularity',variables[i],data=df,color=color1[i],linewidth=3,linestyle='--')

plt.legend(handles=l,bbox_to_anchor=(1,1.43))

plt.title("Trend Analysis Popularity Wise",y=1.15,fontsize=25)

plt.xlabel('Popularity')

plt.ylabel('Values')
tempo = data[['tempo','popularity']]

tempo['popularity']=pd.DataFrame(tempo['popularity']/10).apply(np.floor).astype(int)*10

tempo=pd.DataFrame(tempo.groupby('popularity')['tempo'].mean()).reset_index()

fig = plt.figure(figsize=(15,4))

plt.plot( 'popularity','tempo',data=tempo,color='red',linewidth=1.3)

plt.title("Trend Analysis Popularity Wise - Tempo ",y=1.1,fontsize=25)

plt.xlabel('Popularity')

plt.ylabel('Tempo')
loudness = data[['loudness','popularity']]

loudness['popularity']=pd.DataFrame(loudness['popularity']/10).apply(np.floor).astype(int)*10

loudness=pd.DataFrame(loudness.groupby('popularity')['loudness'].mean()).reset_index()

fig = plt.figure(figsize=(15,4))

plt.plot( 'popularity','loudness',data=loudness,color='#6f4a8e',linewidth=1.4)

plt.title("Trend Analysis Popularity Wise - Loudness ",y=1.1,fontsize=25)

plt.xlabel('Popularity')

plt.ylabel('Loudness (db)')
colors_list = ['#d3dbff','#fe91ca']

mpl.rcParams['font.size'] = 18.0

md = pd.DataFrame(data['mode'].value_counts())

md['mode'].plot(kind='pie',

            figsize=(15, 6),

            autopct='%1.1f%%', 

            startangle=90,  

            shadow=True,       

            labels=None,    

            pctdistance=1.2, 

            colors=colors_list)



plt.title('Mode Ratio',y=1.1,fontsize=25) 

plt.axis('equal') 

plt.legend(labels=['Major','Minor'], loc='upper left') 

plt.show()
key = pd.DataFrame(data['key'].value_counts()).reset_index().sort_values('index')

key.replace({'index' : { 0 : 'C', 1 : 'C#', 2 : 'D', 3 : 'D#', 4 : 'E', 5 : 'F', 6 : 'F#', 

                        7 : 'G', 8 : 'G#', 9 : 'A', 10 : 'A#', 11 : 'B'}} , inplace=True)

fig = plt.figure(figsize=(15,6))

plt.bar( key['index'],

         key['key'],

         width=0.45,

         color = ['#ffa931','#00a8cc'])

plt.title('Frequency Count For Key',y=1.1,fontsize=20)

plt.xlabel('Key')

plt.ylabel('Frequency')

ax.axes.get_xaxis().set_visible(True)
keypop = pd.DataFrame(data.groupby('key')['popularity'].mean()).reset_index()

keypop.replace({'key' : { 0 : 'C', 1 : 'C#', 2 : 'D', 3 : 'D#', 4 : 'E', 5 : 'F', 6 : 'F#', 

                        7 : 'G', 8 : 'G#', 9 : 'A', 10 : 'A#', 11 : 'B'}} , inplace=True)



fig = plt.figure(figsize=(15,6))

plt.bar( keypop['key'],

         keypop['popularity'],

         width=0.45,

         color = ['#844685','#f3c623'])

plt.title('Key VS Popularity',y=1.1,fontsize=20)

plt.xlabel('Key')

plt.ylabel('Popularity')

ax.axes.get_xaxis().set_visible(True)
x=genre_data.sort_values('popularity').tail(10)

fig = plt.figure(figsize=(15,4))

plt.bar( x['genres'],

         x['popularity'],

         width=0.3,

         color = ['#9818d6','#ffa41b'])

plt.title('Genres VS Popularity',y=1.1,fontsize=20)

plt.xlabel('Genres',fontsize=20)

plt.ylabel('Popularity',fontsize=20)

plt.xticks(fontsize=20,rotation=45,ha='right')

ax.axes.get_xaxis().set_visible(True)