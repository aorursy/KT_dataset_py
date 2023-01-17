# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import folium # plotting library

from folium import plugins



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



import matplotlib.pyplot as plt

import matplotlib.cm as cm

import matplotlib.colors as colors



import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/us-weather-events/US_WeatherEvents_2016-2019.csv')



df.head()
df['Type'].value_counts()
df['Severity'].value_counts()
df = df[(df['Severity'] != 'UNK') & (df['Severity'] != 'Other')]



df.head()
df_types = df[['AirportCode','Type']]



df_types.head()
types = pd.get_dummies(df_types['Type'])



types['AirportCode'] = df_types['AirportCode']



types = types.groupby('AirportCode').sum().reset_index()



types.head()
codes = types[['AirportCode']]

types.drop('AirportCode', axis=1, inplace=True)
distortions = []



K = range(1,20)

for k in K:

    kmean = KMeans(n_clusters=k, random_state=0, n_init = 50, max_iter = 500)

    kmean.fit(types)

    distortions.append(kmean.inertia_)
plt.figure(figsize=(10,5))

plt.plot(K, distortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Distortion')

plt.title('The Elbow Method')

plt.show()
# run k-means clustering

kmeans = KMeans(n_clusters=4, random_state=0).fit(types)



codes['cluster'] = kmeans.labels_

codes.head()
pca = PCA().fit(types)

pca_types = pca.transform(types)

print("Variance explained by each component (%): ")

for i in range(len(pca.explained_variance_ratio_)):

      print("\n",i+1,"º:", pca.explained_variance_ratio_[i]*100)

print("Total sum (%): ",sum(pca.explained_variance_ratio_)*100)

print("Explained variance of the first two components (%): ",sum(pca.explained_variance_ratio_[0:1])*100)
c0 = []

c1 = []

c2 = []

c3 = []



for i in range(len(pca_types)):

    if kmeans.labels_[i] == 0:

        c0.append(pca_types[i])

    if kmeans.labels_[i] == 1:

        c1.append(pca_types[i])

    if kmeans.labels_[i] == 2:

        c2.append(pca_types[i])

    if kmeans.labels_[i] == 3:

        c3.append(pca_types[i])

        

        

c0 = np.array(c0)

c1 = np.array(c1)

c2 = np.array(c2)

c3 = np.array(c3)



plt.figure(figsize=(7,7))

plt.scatter(c0[:,0], c0[:,1], c='red', label='Cluster 0')

plt.scatter(c1[:,0], c1[:,1], c='blue', label='Cluster 1')

plt.scatter(c2[:,0], c2[:,1], c='green', label='Cluster 2')

plt.scatter(c3[:,0], c3[:,1], c='black', label='Cluster 3')

plt.legend()

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.title('Low dimensional visualization (PCA) - Airports');
types['cluster']  = kmeans.labels_



types.head()
types.groupby('cluster').mean()
sns.catplot(x='cluster', y='Cold', data=types, kind='bar');
sns.catplot(x='cluster', y='Fog', data=types, kind='bar');
sns.catplot(x='cluster', y='Rain', data=types, kind='bar');
sns.catplot(x='cluster', y='Snow', data=types, kind='bar');
sns.catplot(x='cluster', y='Storm', data=types, kind='bar');
latitude = 38.500000

longitude = -95.665



map_USA = folium.Map(location=[latitude, longitude], zoom_start=4)



map_USA
airports = df[['AirportCode', 'LocationLat','LocationLng','City','State']]



airports.head()
number_of_occurences = pd.DataFrame(airports['AirportCode'].value_counts())

number_of_occurences.reset_index(inplace=True)

number_of_occurences.columns = ['AirportCode', 'Count']

number_of_occurences.head()
number_of_occurences = number_of_occurences.merge(airports.drop_duplicates())



number_of_occurences = number_of_occurences.merge(codes)



number_of_occurences.head()
occurences = folium.map.FeatureGroup()

n_mean = number_of_occurences['Count'].mean()



for lat, lng, number, city, state in zip(number_of_occurences['LocationLat'],

                                         number_of_occurences['LocationLng'],

                                         number_of_occurences['Count'],

                                         number_of_occurences['City'],

                                         number_of_occurences['State'],):

    occurences.add_child(

        folium.vector_layers.CircleMarker(

            [lat, lng],

            radius=number/n_mean*5, # define how big you want the circle markers to be

            color='yellow',

            fill=True,

            fill_color='blue',

            fill_opacity=0.6,

            tooltip = str(number)+','+str(city) +','+ str(state)

        )

    )



map_USA.add_child(occurences)
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=4)



# set color scheme for the clusters

x = np.arange(4)

ys = [i + x + (i*x)**2 for i in range(4)]

colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))

rainbow = [colors.rgb2hex(i) for i in colors_array]



# add markers to the map

markers_colors = []

for lat, lng, cluster, city, state in zip(number_of_occurences['LocationLat'], number_of_occurences['LocationLng'],  

                                            number_of_occurences['cluster'],

                                         number_of_occurences['City'],

                                         number_of_occurences['State']):

    #label = folium.Popup(str(city)+ ','+str(state) + '- Cluster ' + str(cluster), parse_html=True)

    folium.vector_layers.CircleMarker(

        [lat, lng],

        radius=5,

        #popup=label,

        tooltip = str(city)+ ','+str(state) + '- Cluster ' + str(cluster),

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.9).add_to(map_clusters)

       

map_clusters