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
import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import folium

from sklearn.cluster import KMeans
dataframe = pd.read_csv('/kaggle/input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv') 
dataframe.head(5)
dataframe.info()
sns.countplot(dataframe['Gender'])
sns.countplot(dataframe['Race'],hue=dataframe['Gender'])
sns.countplot(dataframe['status'])
dataframe['status'].value_counts()
dataframe['Race'].value_counts().plot(kind='bar')
longitude = dataframe['lat'].values

latitude = dataframe['lon'].values
m = folium.Map(location=[34.0593, -118.274])
for i in range(63):

    if (i!= 51):

        folium.Marker([latitude[i],longitude[i]],popup = dataframe.loc[i,'Full Name'] ).add_to(m)

m
values = {'lat': 34.0593 , 'lon': -118.274} 

filedata = dataframe.fillna(value=values)
def get_kmeans_score(data, center):

  

    kmeans = KMeans(n_clusters=center)

    model = kmeans.fit(data)

    score = np.abs(model.score(data))

    

    return score



scores = []

centers = list(range(1,11))



for center in centers:

    scores.append(get_kmeans_score(filedata.loc[: ,['lon','lat']], center))

    

plt.plot(centers, scores, linestyle='--', marker='o', color='b');

plt.xlabel('Number of Centers');

plt.ylabel('SSE');

plt.title('SSE vs. Number');
kmeans = KMeans(n_clusters=2)



kmeans.fit(filedata.loc[: ,['lon','lat']]).cluster_centers_
folium.Marker([34.02756075,-118.28079299],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

folium.Marker([34.0593,-118.274],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)



m