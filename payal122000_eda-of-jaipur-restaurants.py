# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np 
import pandas as pd
import os
import seaborn as sns
print(os.listdir("../input"))
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly.offline as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=False)
from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from folium.plugins import HeatMap
import folium
from tqdm import tqdm
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.colors as mcolors
from sklearn.manifold import TSNE
from gensim.models import word2vec
import nltk
df=pd.read_csv("../input/restaurants-in-jaipur/Cleaned.csv")
print("dataset contains {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.info()
df.head()
#Exploratory data analysis

#which are the top restaurant chains in jaipur

plt.figure(figsize=(10,7))
chains=df['RestaurantName'].value_counts()[:20]
sns.barplot(x=chains,y=chains.index,palette='deep')
plt.title("Most famous restaurants chains in Jaipur")
plt.xlabel("Number of outlets")
# cost_dist=df[['Ratings','CostForTwo']].dropna()
# cost_dist['Ratings']=cost_dist['Ratings'].apply(lambda x: float(x.split('/')[0]) if len(int(x))>3 else 0)
# cost_dist['CostForTwo']=cost_dist['CostForTwo'].apply(lambda x: int(x.replace(',','')))
# plt.figure(figsize=(10,7))
# sns.scatterplot(x="Ratings",y='CostForTwo',data=cost_dist)
# plt.show()
print(df['Ratings'].dropna())
rateDist = df['Ratings'].dropna()
ax =sns.distplot(rateDist, hist=True, bins=30, color='#15b01a')
plt.title("Distribution of Ratings")
#Distribution of cost for two people

plt.figure(figsize=(6,6))
sns.distplot(cost_dist['CostForTwo'])
plt.show()
#which are the most common restaurant type in jaipur?

plt.figure(figsize=(7,7))
rest=df['Category'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")
#Cost factor ?

trace0=go.Box(y=df['CostForTwo'],
              marker = dict(
        color = 'rgb(214, 12, 140)',
    ))
data=[trace0]
layout=go.Layout(title="Box plot of approximate cost",width=800,height=500,yaxis=dict(title="Price"))
fig=go.Figure(data=data,layout=layout)
py.iplot(fig)

#Which are the foodie areas?

plt.figure(figsize=(7,7))
Rest_locations=df['Locality'].value_counts()[:20]
sns.barplot(Rest_locations,Rest_locations.index,palette="rocket")
#Which are the most common cuisines in each locations?

df_1=df.groupby(['Locality','Cuisines']).agg('count')
data=df_1.sort_values(['Ratings'],ascending=False).groupby(['Locality'],
                as_index=False).apply(lambda x : x.sort_values(by="Ratings",ascending=False).head(3))['Ratings'].reset_index().rename(columns={'Ratings':'count'})

data.head(10)
locations=pd.DataFrame({"RestaurantName":df['Locality'].unique()})
locations['RestaurantName']=locations['RestaurantName'].apply(lambda x: "Jaipur " + str(x))
lat_lon=[]
geolocator=Nominatim(user_agent="app")
for location in locations['RestaurantName']:
    location = geolocator.geocode(location)
    if location is None:
        lat_lon.append(np.nan)
    else:    
        geo=(location.latitude,location.longitude)
        lat_lon.append(geo)


locations['geo_loc']=lat_lon
locations.to_csv('locations.csv',index=False)
locations["RestaurantName"]=locations['RestaurantName'].apply(lambda x :  x.replace("Jaipur","")[1:])
locations.head()
Rest_locations=pd.DataFrame(df['Locality'].value_counts().reset_index())
Rest_locations.columns=['RestaurantName','count']
Rest_locations=Rest_locations.merge(locations,on='RestaurantName',how="left").dropna()
Rest_locations['count'].max()
def generateBaseMap(default_location=[26.91, 75.78], default_zoom_start=12):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start)
    return base_map
lat,lon=zip(*np.array(Rest_locations['geo_loc']))
Rest_locations['lat']=lat
Rest_locations['lon']=lon
basemap=generateBaseMap()
HeatMap(Rest_locations[['lat','lon','count']].values.tolist(),zoom=20,radius=15).add_to(basemap)
#Heatmap of restaurant count on each location
basemap
