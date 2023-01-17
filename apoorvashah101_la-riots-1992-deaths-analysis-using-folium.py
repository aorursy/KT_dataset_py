# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

#importing the data
filename = '../input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv'
data = pd.read_csv(filename)
#Exploring the data
data.info()
data.head()


#Here I am switching latitude and longitude because it is incorrectly labeled in the dataset.
longitude = data['lat'].values
latitude = data['lon'].values

m = folium.Map(location=[34.0593, -118.274], zoom_start = 10)

# I skipped 51 because that one showed up as a NaN.
for i in range(63):
    if (i != 51):
        folium.Marker([latitude[i],longitude[i]],popup = data.loc[i,'Full Name'] ).add_to(m)
m        
plt.plot(data['lat'], data['lon'],'.')
plt.title('Plot of Deaths')
plt.show()
#I decided to just fill in the missing value for the following kmeans clustering.
values = {'lat': 34.0593, 'lon':-118.274}
filleddata = data.fillna(value=values)
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
  
    kmeans = KMeans(n_clusters=center)
    model = kmeans.fit(data)
    score = np.abs(model.score(data))
    
    return score

scores = []
centers = list(range(1,11))

for center in centers:
    scores.append(get_kmeans_score(filleddata.loc[: ,['lon','lat']], center))
    
plt.plot(centers, scores, linestyle='--', marker='o', color='b');
plt.xlabel('Number of Centers');
plt.ylabel('SSE');
plt.title('SSE vs. Number');
kmeans = KMeans(n_clusters=2)

kmeans.fit(filleddata.loc[: ,['lon','lat']]).cluster_centers_
#I am not sure why it's switching the coordinates for the first cluster

#Plotting the two centers 
folium.Marker([34.02756075,-118.28079299],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)
folium.Marker([34.0593,-118.274],popup = 'Cluster Center', icon=folium.Icon(color='red') ).add_to(m)

m