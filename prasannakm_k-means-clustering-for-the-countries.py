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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
data = pd.read_csv("/kaggle/input/world-foodfeed-production/FAO.csv",encoding="ISO-8859-1")
data.head()
print("Dataset contains",data.shape[0],"observations and" , data.shape[1], "attributes")
data.columns
data.describe()
data.info()
cat_cols = data.select_dtypes(['object']).columns
print("Count of categorical values are",cat_cols.value_counts().sum())
num_cols = data.select_dtypes(['float64','int64']).columns
print("Count of numerical values are",num_cols.value_counts().sum())
cols = list(num_cols)
import seaborn as sns
for col in cols:
    sns.boxplot(y = data[col])
    plt.show()
pd.set_option('display.max_rows',None)
data.isnull().sum()
fig = plt.subplots(figsize=(20,10))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
#for col in data.columns:
    #if (col not in ['Area Abbreviation','Area Code','Area','Item Code','Item','Element Code','Element','Unit','latitude','longitude'])& (data[col].isnull().sum()>0):
        #data.loc[data[col].isnull(),col]=data[col].median()
data = data.dropna()
fig = plt.subplots(figsize=(20,10))
sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()
data['Y2012'] = data['Y2012'].replace('-','')
data['Y2013'] = data['Y2013'].replace('-','')
data['Total_production'] = (data['Y1961'] + data['Y1962'] + data['Y1963'] + data['Y1964'] + data['Y1965'] + data['Y1966'] + 
    data['Y1967'] + data['Y1968'] + data['Y1969'] + data['Y1970'] + data['Y1971'] + data['Y1972'] + data['Y1973'] +
    data['Y1974'] + data['Y1975'] + data['Y1976'] + data['Y1977'] + data['Y1978'] + data['Y1979'] + data['Y1980'] + 
    data['Y1981'] + data['Y1982'] + data['Y1983'] + data['Y1984'] + data['Y1985'] + data['Y1986'] + data['Y1987'] + 
    data['Y1988'] + data['Y1989'] + data['Y1990'] + data['Y1991'] + data['Y1992'] + data['Y1993'] + data['Y1994'] + 
    data['Y1995'] + data['Y1996'] + data['Y1997'] + data['Y1998'] + data['Y1999'] + data['Y2000'] + data['Y2001'] + 
    data['Y2001'] + data['Y2002'] + data['Y2003'] + data['Y2004'] + data['Y2005'] + data['Y2006'] + data['Y2007'] + 
    data['Y2008'] + data['Y2009'] + data['Y2010'] + data['Y2011'] + data['Y2012'] + data['Y2013'] )
fig = plt.subplots(figsize=(7,7))
ax = sns.countplot(data['Element'],order=data['Element'].value_counts().index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
df_food = data[data['Element']=='Food']
df_food.shape
df_food.head()
df_food.groupby('Item').sum()['Total_production'].sort_values(ascending=False)[:10]
fig = plt.subplots(figsize=(10,10))
food_item = df_food.groupby('Item').sum()['Total_production'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = df_food,x = food_item.index, y= food_item.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
fig = plt.subplots(figsize=(12,10))
Area_wise_food = df_food.groupby('Area').sum()['Total_production'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = df_food,x = Area_wise_food.index, y= Area_wise_food.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
df_feed = data[data['Element']=='Feed']
df_feed.head()
fig = plt.subplots(figsize=(12,10))
feed_item = df_feed.groupby('Item').sum()['Total_production'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = df_feed,x = feed_item.index, y= feed_item.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
fig = plt.subplots(figsize=(12,10))
Area_wise_feed = df_feed.groupby('Area').sum()['Total_production'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = df_feed,x = Area_wise_feed.index, y= Area_wise_feed.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
fig = plt.subplots(figsize=(12,10))
Area_wise = data.groupby('Area').sum()['Total_production'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = data,x = Area_wise.index, y= Area_wise.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.1f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()
sns.factorplot('Area',data=data[(data['Area']=='China, mainland')|(data['Area']=='United States of America')|(data['Area']=='India')],kind='count',hue='Element')
plt.show()
data = data.drop('Total_production', axis=1)
data.head(2)
#Pivoting some columns
columns = ['Area Abbreviation', 'Area Code', 'Area', 'Item Code', 'Item','Element Code', 
           'Element', 'Unit', 'latitude', 'longitude']
df = pd.melt(data,id_vars= columns)
df.drop(columns=['Area Code','Item Code','Area Abbreviation','Unit','Element Code'], axis=1,inplace=True)
df.rename(str.lower, axis = 1, inplace = True)
df.rename({'variable':'year','value':'quantity','area':'country'},axis=1,inplace=True)
# Removing the Y from the numbers in df.year
df.year = df.year.str.replace('Y','')
df.country = df.country.replace ({'China, mainland': 'China','United States of America':'USA',
                                 'United Kingdom':'UK'})
df.head(2)
df.info()
df['year'] = pd.to_datetime(df['year'],infer_datetime_format=True)
df1 = df[(df['country']=='China')|(df['country']=='USA')|(df['country']=='India')][['item','element','latitude','longitude','year','quantity','country']]
fig = plt.subplots(figsize=(10,10))
sns.lineplot(x='year', y='quantity', data=df1,hue='country',style='country',markers=True)
fig = plt.subplots(figsize=(10,10))
sns.lineplot(x='year', y='quantity', data=df,hue='element',style='element',markers=True)
data.head()
df_countries = data.copy()
df_countries.head()
df_countries.columns
df_countries = df_countries.drop(['Area Abbreviation', 'Area Code','Item Code', 'Item',
       'Element Code', 'Element', 'Unit', 'latitude', 'longitude'],axis=1)
df_countries.head()
df_countries = df_countries.groupby('Area').sum()
df_countries.head()
df_countries.shape
#sns.pairplot(df_countries,diag_kind='kde',hue='Area')
df_scaled = df_countries.apply(zscore)
df_scaled.head()
df_scaled.shape
cluster_range = range(1,15)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters,n_init = 10)
    clusters.fit(df_scaled)
    cluster_errors.append(clusters.inertia_)
clusters_df = pd.DataFrame({"num_clusters":cluster_range, "cluster_errors": cluster_errors})
clusters_df
plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
#silhouette analysis
from sklearn.metrics import silhouette_score
score = []
for n_clusters in range(2,20):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score.append(silhouette_score(df_scaled,labels,metric='euclidean'))
    
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(score)
plt.grid(True)
plt.ylabel('silhouette score')
plt.xlabel('k')
plt.title('silhouette for kmeans')
kmeans = KMeans(n_clusters=4, n_init = 15, random_state=25)
kmeans.fit(df_scaled)
centroids = kmeans.cluster_centers_
centroids
centroid_df = pd.DataFrame(centroids, columns = list(df_scaled) )
centroid_df
kmeans.inertia_
## creating a new dataframe only for labels and converting it into categorical variable
df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))

df_labels['labels'] = df_labels['labels'].astype('category')

df_labels
df_countries['labels'] =kmeans.labels_
df_countries.head()
df_countries['labels'].value_counts()
df_countries.tail(8)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=100)
kmeans.fit(df_scaled)
labels = kmeans.labels_
ax.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], df_scaled.iloc[:, 3],c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans Clustering')
kmeans = KMeans(n_clusters=2, n_init = 15, random_state=25)
kmeans.fit(df_scaled)
centroids = kmeans.cluster_centers_
centroids
centroid_df = pd.DataFrame(centroids, columns = list(df_scaled) )
centroid_df
kmeans.inertia_
## creating a new dataframe only for labels and converting it into categorical variable
df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))

df_labels['labels'] = df_labels['labels'].astype('category')

df_labels
df_countries['labels'] =kmeans.labels_
df_countries.head()
df_countries['labels'].value_counts()
df_countries.tail(8)
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=100)
kmeans.fit(df_scaled)
labels = kmeans.labels_
ax.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], df_scaled.iloc[:, 3],c=labels.astype(np.float), edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Length')
ax.set_ylabel('Height')
ax.set_zlabel('Weight')
ax.set_title('3D plot of KMeans Clustering')
#silhouette analysis
from sklearn.metrics import silhouette_score
score = []
for n_clusters in range(2,20):
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score.append(silhouette_score(df_scaled,labels,metric='euclidean'))
    
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(score)
plt.grid(True)
plt.ylabel('silhouette score')
plt.xlabel('k')
plt.title('silhouette for kmeans')
centroid_df