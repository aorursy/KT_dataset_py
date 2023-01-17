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
# Dowmload Dataframe

df=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df.head()
df.info()
df1=df.drop(columns=['id', 'name', 'host_id', 'host_name','last_review','reviews_per_month'])

# df1=df1.iloc[10000,:]

df1.columns
# Split data into 2 Dataset

from sklearn.model_selection import train_test_split

x_train, x_test = train_test_split(df1, test_size=0.5, random_state=42)

x_train.shape,x_test.shape
#plot correlation

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm 

cols=[ 'price', 'minimum_nights', 'number_of_reviews',

       'calculated_host_listings_count', 'availability_365']

cm = np.corrcoef(x_train[cols].values.T) 

f, ax = plt.subplots(figsize =(6,4)) 

  

sns.heatmap(cm, ax = ax, cmap ="YlGnBu", 

            linewidths = 0.1, yticklabels = cols,  

                              xticklabels = cols, annot=True, fmt=".2f") 
x_train.describe()
xtrain1=pd.get_dummies(x_train,columns=['neighbourhood_group', 'neighbourhood','room_type', ])

xtrain1.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

xtrain_tr = scaler.fit_transform(xtrain1)
from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

from sklearn import metrics

from sklearn.cluster import KMeans
reduced_data = PCA(n_components=2).fit_transform(xtrain_tr )
xtrain_tr.shape[1]
# Bulid dataframe

d=pd.DataFrame(x_train,columns=['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude',

       'room_type', 'price', 'minimum_nights', 'number_of_reviews',

       'calculated_host_listings_count', 'availability_365'])

d.head()
import matplotlib.pyplot as plt

sse = {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(reduced_data)

    c="clusters_"+str(k)

    #labels_:Labels of each point

    d[c] = kmeans.labels_

    # inertia_: Sum of distances of samples to their closest cluster center

    sse[k] = kmeans.inertia_ 

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(reduced_data, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Customers')

plt.ylabel('Euclidean distances')

plt.show()
import matplotlib.pyplot as plt



hc = KMeans(n_clusters=3, max_iter=1000)

y_hc = hc.fit_predict(reduced_data)



# Visualising the clusters

plt.scatter(reduced_data[y_hc == 0, 0], reduced_data[y_hc == 0, 1], s = 5, c = 'blue', label = 'Cluster 0')

plt.scatter(reduced_data[y_hc == 1, 0], reduced_data[y_hc == 1, 1], s = 5, c = 'orange', label = 'Cluster 1')

plt.scatter(reduced_data[y_hc == 2, 0], reduced_data[y_hc == 2, 1], s = 5, c = 'green', label = 'Cluster 2')

plt.title('Clusters of customers')

plt.xlabel('reduced_data[:, 0]')

plt.ylabel('reduced_data[:, 1]')

plt.legend()

plt.show()
d.head(3)
g=sns.boxplot(x="clusters_3", y="price", data=d)

g.set(ylim=(-10, 600))
p=sns.boxplot(x="clusters_3", y="minimum_nights", data=d)

p.set(ylim=(-5, 20))
sns.boxplot(x="clusters_3", y="availability_365", data=d)
acc2 = d.groupby(['clusters_3','neighbourhood_group'])['neighbourhood'].count()

acc2=acc2.reset_index()

acc2
import numpy as np

from matplotlib import rc

import pandas as pd

 

# Data

r = [0,1,2]

raw_data = {'Brooklyn': [10100, 5,0], 'Queens': [1, 0, 2747],'Staten Island': [188,1,0],

           'Manhattan': [0,10860,0],'Bronx': [0,0,545]}

dat = pd.DataFrame(raw_data)

 

# From raw value to percentage

totals = [i+j+k+l+f for i,j,k,l,f in zip(dat['Brooklyn'], dat['Queens'], dat['Staten Island'], 

                                         dat['Manhattan'], dat['Bronx'])]

Brooklyn = [i / j * 100 for i,j in zip(dat['Brooklyn'], totals)]

Manhattan = [i / j * 100 for i,j in zip(dat['Manhattan'], totals)]

Queens = [i / j * 100 for i,j in zip(dat['Queens'], totals)]

StatenIsland = [i / j * 100 for i,j in zip(dat['Staten Island'], totals)]

Bronx = [i / j * 100 for i,j in zip(dat['Bronx'], totals)]



# plot

barWidth = 0.85

names = ('0','1','2')

plt.bar(r, Brooklyn, color='#0A709A', edgecolor='white', width=barWidth,label="Brooklyn")

plt.bar(r, Manhattan, bottom=Brooklyn, color='#ff6600', edgecolor='white', width=barWidth,label='Manhattan')

plt.bar(r, Queens, bottom=[i+j for i,j in zip(Brooklyn, Manhattan)], color='#1f7a1f',

        edgecolor='white', width=barWidth,label='Queens')

plt.bar(r, StatenIsland, bottom=[i+j+k for i,j,k in zip(Brooklyn, Manhattan,Queens)],

        color='#b5ffb9', edgecolor='white', width=barWidth,label='Staten Island')

plt.bar(r, Bronx, bottom=[i+j+k+l for i,j,k,l in zip(Brooklyn, Manhattan,StatenIsland,Queens)], 

        color= '#FDDB5E', edgecolor='white', width=barWidth,label='Bronx')



plt.xticks(r, names)

plt.ylabel("neighbourhood_group")

plt.xlabel("group")

plt.legend(loc='best', bbox_to_anchor=(1,1), ncol=1)

plt.show()
fplot, non_fplot = train_test_split(d, test_size=0.97, random_state=42)

fplot.shape,non_fplot.shape
from folium import plugins

import folium



# define latitude and longitude of new york city.

new_york_map = folium.Map(location=[40.730610, -73.935242], zoom_start=12)





# label each point.

for lat, lng, label, in zip(fplot.latitude, fplot.longitude,fplot.clusters_3):

#   cluster 0 is blue color

    if label==0:

        clo='blue'    

        

#   cluster 1 is orange color

    elif label==1 :

        clo='orange'

        

#   cluster 2 is green color

    elif label==2 :

        clo='green'

    folium.Marker(

            location=[lat, lng],

            icon=folium.Icon(color=clo),

            popup=label,

        ).add_to(new_york_map)

incidents_accident = folium.map.FeatureGroup()

new_york_map.add_child(incidents_accident)

new_york_map