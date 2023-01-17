import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/forest-fires-in-brazil/amazon.csv",encoding='latin1') 
data.head() 
data.isna().sum()
data.duplicated().sum()
data.drop_duplicates(inplace=True) 
data.drop('date',axis=1,inplace =True) 
data = data.reset_index(drop=True)
data = data.replace({'Piau':'Piaui',

                     'Rio':'Rio de Janeiro'})
plt.figure(figsize=(15,10))

ax=sns.distplot(data.number)

ax = ax.set(yticklabels=[],title='Histogram of number of fires reported')
ax = plt.figure(figsize=(15,10))



plt.subplot(2,1,1)

ax = sns.boxplot(data.year,data.number)



plt.subplot(2,1,2)

ax = sns.boxplot(data.state,data.number)

ax = plt.xticks(rotation=65)
pd.DataFrame(data.groupby(data.year).number.std())
pd.DataFrame(data.groupby(data.state).number.std())
x = data.groupby(data.year).number.median()



ax = plt.figure(figsize=(15,10))

ax = plt.plot(x.index.values,x.values) 

ax = plt.title('Median of number of fires reported')

ax = plt.xlabel('Year')

ax = plt.ylabel('Median')



z = np.polyfit(x.index.values, x.values, 1)

p = np.poly1d(z)

ax = plt.plot(x.index.values,p(x.index.values),"r--")



ax = plt.legend(['Real Data','Trend Line'])
x = data.groupby(data.month, sort=False).number.median()

plt.figure(figsize=(15,10))

ax = plt.plot(x.index.values, x.values) 

ax = plt.xticks(rotation=65)

ax = plt.title('Median of number of fires reported')

ax = plt.xlabel('Month')

ax = plt.ylabel('Median')
x = data.groupby(data.state).number.median()

plt.figure(figsize=(15,10))

ax = sns.barplot(y=x.index.values,x=x.values)

ax = ax.set(xlabel='Median of fires reported',ylabel='States',title='Fires Reported by State')
import geopandas as gpd
map_brazil = gpd.read_file('../input/brazil-geopandas-data/gadm36_BRA_1.shp')

map_brazil = map_brazil[['NAME_1','geometry']]

map_brazil = map_brazil.to_crs(epsg=4326)
map_brazil.head()
import unidecode

map_brazil['NAME_1'] = map_brazil['NAME_1'].apply(lambda x: unidecode.unidecode(x))

data['state'] = data['state'].apply(lambda x: unidecode.unidecode(x))
map_brazil['centroid'] = map_brazil.geometry.centroid
median = data.groupby('state').number.median()



map_brazil = map_brazil.join(median,on='NAME_1')
map_brazil.head()
data['month'] = data.month.replace(data.month.unique(),range(1,13))
fig,ax = plt.subplots(figsize=(20,10))

map_brazil.plot(column='number',ax=ax,alpha=0.4,edgecolor='black',cmap='YlOrRd',legend=True)

plt.title("Median Number of Fires")

plt.axis('off')





for x, y, label in zip(map_brazil.centroid.x, map_brazil.centroid.y, map_brazil.NAME_1):

    ax.annotate(label, xy=(x, y), xytext=(3,3), textcoords="offset points",color='blue')
map_brazil['lat_long'] = map_brazil.centroid.apply(lambda x : x.coords[0])
x = map_brazil[['NAME_1','lat_long']]

x = x.set_index('NAME_1')
data = data.set_index('state')
data = data.join(x)
data[['x', 'y']] = pd.DataFrame(data['lat_long'].tolist(), index=data.index) 

data.drop('lat_long',axis=1,inplace=True)
from sklearn.cluster import KMeans
inertia =[] 

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)

    inertia.append(kmeans.inertia_)
ax = plt.plot(range(1,10),inertia)
kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
from sklearn.preprocessing import StandardScaler



x = StandardScaler().fit_transform(data)



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['1', '2'])



plt.figure(figsize=(10,10))

ax = plt.scatter(principalDf['1'],principalDf['2'],c=kmeans.labels_)
pc = pd.DataFrame(pca.components_) 

pc.columns = data.columns 

pc
data['labels'] = kmeans.labels_
data.labels.value_counts() 
plt.figure(figsize=(10,10))

ax = sns.boxplot(data.labels,data.number)
plt.figure(figsize=(10, 10))

ax = sns.countplot(data.month,hue=data.labels)
plt.figure(figsize=(10, 10))

ax = sns.countplot(data.year,hue=data.labels)
plt.figure(figsize=(10, 10))

ax = sns.countplot(data.index.values,hue=data.labels)

ax = plt.xticks(rotation=65)