# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import folium

from wordcloud import WordCloud, STOPWORDS 

from sklearn.cluster import KMeans

sns.set_style("darkgrid")

df = pd.read_csv("/kaggle/input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv")

df.head()
df.isnull().sum()


df = df.drop(columns=['URL'],axis=1)
import plotly.express as px
px.histogram(df['Age'],opacity=0.8,color_discrete_sequence=['indianred'])
df['Race'].value_counts()
df['Race'] = df['Race'].replace({" Latino":"Latino"," White":"White"," Asian":"Asian"," Black":"Black"})

sns.countplot(df['Race'])
df = df.replace({" Male":"Male"," Female":"Female"})
sns.countplot(df['Gender'])
df['Gender'].value_counts()
fig = sns.countplot(df['status'])

fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.figure(figsize=(20,8))

fig = sns.countplot(df['Neighborhood'])

fig.set_xticklabels(fig.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()

# LETS SEE THE MOST PROMINENT STORIES

comment_words = '' 

stopwords = set(STOPWORDS) 

for val in df['Story']: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1200, height = 800, 

                background_color ='red', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                        

plt.figure(figsize = (16, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
# PEOPLE 

comment_words = '' 

stopwords = set(STOPWORDS) 

for val in df['Full Name']: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1200, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                        

plt.figure(figsize = (16, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 
comment_words = '' 

stopwords = set(STOPWORDS) 

for val in df['Map Description']: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1600, height = 800, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                        

plt.figure(figsize = (12, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 
comment_words = '' 

stopwords = set(STOPWORDS) 

for val in df['Address']: 

      

    # typecaste each val to string 

    val = str(val) 

  

    # split the value 

    tokens = val.split() 

      

    # Converts each token into lowercase 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

      

    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1600, height = 800, 

                background_color ='black', 

                stopwords = stopwords, 

                min_font_size = 10).generate(comment_words)

# plot the WordCloud image                        

plt.figure(figsize = (12, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

  

plt.show() 


longitude = df['lat'].values

latitude = df['lon'].values
m = folium.Map(location=[34.0593, -118.274], zoom_start = 10) # LA

for i in range(63):

    if (i != 51):

        folium.Marker([latitude[i],longitude[i]],popup = df.loc[i,'Full Name'] ).add_to(m)

m  