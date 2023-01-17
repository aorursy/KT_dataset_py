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
from sklearn.cluster import KMeans

from sklearn.manifold import TSNE

from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration

from sklearn.feature_extraction.text import CountVectorizer

import sklearn.metrics as sm

from sklearn.preprocessing import OneHotEncoder, LabelEncoder,StandardScaler, normalize

from sklearn.impute import SimpleImputer

from scipy.cluster.hierarchy import dendrogram, cophenet

from scipy.cluster.hierarchy import linkage , fcluster

from scipy.spatial import distance

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from matplotlib import pyplot as plt

from sklearn.datasets.samples_generator import make_blobs

from sklearn.metrics import pairwise_distances_argmin

from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.model_selection import KFold

from sklearn.mixture import GaussianMixture



# for Box-Cox Transformation

from scipy import stats

from mlxtend.preprocessing import minmax_scaling



import seaborn as seabornInstance 

from pylab import rcParams

import seaborn as sns

from collections import OrderedDict

cmaps = OrderedDict()

#sns.set(style="darkgrid")

import random

import glob

import copy

from sklearn import preprocessing

from sklearn.preprocessing import scale
data = pd.read_csv('../input/covid19-demographic-predictors/covid19_by_country.csv')

data_2 = data.copy()
data.columns.values
#show data

data.info()

data.shape
data.ndim
count_by_column = (data.sum())

print(count_by_column)
missing_val_count_by_column = (data.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
data.replace([np.inf, - np.inf], np.nan)

data.isnull().any()

new_tests = data['Tests'].mean()

new_tests
def impute_annual_inc(x):

    if pd.isnull(x):

        return new_tests

    else:

        return x

    

data['Tests'] = data['Tests'].apply(impute_annual_inc)

new_top = data['Test Pop'].mean()

new_top
def impute_annual_inc(x):

    if pd.isnull(x):

        return new_top

    else:

        return x

    

data['Test Pop'] = data['Test Pop'].apply(impute_annual_inc)
new_sex_ratio = data['Sex Ratio'].mean()

new_sex_ratio
def impute_annual_inc(x):

    if pd.isnull(x):

        return new_sex_ratio

    else:

        return x

    

data['Sex Ratio'] = data['Sex Ratio'].apply(impute_annual_inc)
#drop ข้อมูลที่เกิด missing มากที่สุดออก



new_data = data.drop(data.columns[[5,6,7,8]], axis=True).head(10)

new_data

new_data.isnull().any()
#หาค่าเฉลยของ column ที่เราจะใช้ในการ clustering ต่อไป

new_data.describe()
#Qua_data= pd.get_dummies(new_data['Quarantine'])



categories = ['Tests', 'Test Pop', 'Country', 'Quarantine', 'Schools', 'Restrictions']



get_numeric_data = pd.get_dummies(data.copy(), columns=categories, drop_first=True)
get_numeric_data.head()
df = data.groupby(["Country"])[['Tests', 'Country', 'Total Infected', 'Total Deaths', 'Total Recovered']].sum().reset_index()

sorted_By_Confirmed=df.sort_values('Total Infected',ascending=False)

sorted_By_Confirmed=sorted_By_Confirmed.drop_duplicates('Country')
Recovered_rate=(sorted_By_Confirmed['Total Recovered']*100)/sorted_By_Confirmed['Total Infected']

Deaths_rate=(sorted_By_Confirmed['Total Deaths']*100)/sorted_By_Confirmed['Total Infected']

#cases_rate=(sorted_By_Confirmed.Confirmed*100)/world_Confirmed_Total



sorted_By_Confirmed['Active']=sorted_By_Confirmed['Total Infected']-sorted_By_Confirmed['Total Deaths']-sorted_By_Confirmed['Total Recovered']

sorted_By_Confirmed['Recovered Cases Rate %']=pd.DataFrame(Recovered_rate)

sorted_By_Confirmed['Deaths Cases Rate %']=pd.DataFrame(Deaths_rate)

#sorted_By_Confirmed['Total Cases Rate %']=pd.DataFrame(cases_rate)





print("Sorted By Total Infected Cases")

sorted_By_Confirmed.style.background_gradient(cmap="tab20b")
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(15,15 ))



sns.barplot(x="Total Infected", y="Country", data=sorted_By_Confirmed.head(20),

            label="Total Infected", color="r")



# Plot the crashes where alcohol was involved

sns.set_color_codes("muted")

sns.barplot(x="Total Recovered", y="Country", data=sorted_By_Confirmed.head(20),

            label="Total Recovered", color="g")



sns.set_color_codes("muted")

sns.barplot(x="Total Deaths", y="Country", data=sorted_By_Confirmed.head(20),

            label="Total Deaths", color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)
x=sorted_By_Confirmed.Country.head(10)

y=sorted_By_Confirmed.Tests

plt.rcParams['figure.figsize'] = (12, 10)

sns.barplot(x,y,order=x ,palette="tab20").set_title('Total Infected / Total Deaths / Total Recovered')  #graf çizdir (Most popular)
sns.pairplot(new_data, diag_kind="kde", markers="+",

                 plot_kws=dict(s=50, edgecolor="b", linewidth=1),

                 diag_kws=dict(shade=True))
X = new_data.iloc[:, [3,4]].values #.iloc for positional indexing

y = new_data.iloc[:,].values
# Calculate the linkage: mergings

mergings = linkage(X, method='complete')



# Plot the dendrogram, using varieties as labels

dendrogram(mergings,

           labels=X,

           leaf_rotation=90,

           leaf_font_size=12,

)



plt.title("Hirerachy Clustrring method 'Complete'")

plt.xlabel('Cluster size')

plt.ylabel('number')

plt.show()
linked = linkage(X, 'single')



plt.figure(figsize=(10, 8))

dendrogram(linked, truncate_mode='lastp', p=15, 

           leaf_rotation=45, leaf_font_size=15,

            show_contracted=True)



plt.title("Hirerachy Clustrring method 'Single'")

plt.xlabel('Cluster size')

plt.ylabel('number')



plt.show()
normalized_movements = normalize(X)



# Calculate the linkage: mergings

mergings = linkage(normalized_movements, method='complete')



# Plot the dendrogram

dendrogram(mergings,

           labels=X,

           leaf_rotation=90,

           leaf_font_size=12,

)



plt.title("Hirerachy Clustrring method 'Complete', Normalize")

plt.xlabel('Cluster size')

plt.ylabel('number')

plt.show()
#Generate hierarchical cluters

k = 2



Hclustering = FeatureAgglomeration(n_clusters=k, affinity='euclidean', linkage='ward')
Hclustering.fit(X)



Hclustering.labels_
#plot graph



plt.figure(figsize=(10,10))

plt.title("Scatter plot")

plt.scatter(X[:,0], X[:, 1])
#set random seed



random.seed(30)
# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []

for cluster in range(1,20):

    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')

    kmeans.fit(get_numeric_data)

    SSE.append(kmeans.inertia_)



# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})

plt.figure(figsize=(12,6))

plt.plot(frame['Cluster'], frame['SSE'], marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')
model = KMeans(n_clusters = 4)

model.fit(get_numeric_data)
labels = model.predict(get_numeric_data)

print(labels)
km = KMeans(

    n_clusters=4, init='random',

    n_init=10, max_iter=300, 

    tol=1e-04

)



  



y_km = km.fit_predict(X)





plt.scatter(

    X[y_km == 0, 0], X[y_km == 0, 1],

    s=150, c='lightgreen',

    label='cluster 1'

)



plt.scatter(

    X[y_km == 1, 0], X[y_km == 1, 1],

    s=150, c='orange',

    label='cluster 2'

)



plt.scatter(

    X[y_km == 2, 0], X[y_km == 2, 1],

    s=150, c='lightblue',

    label='cluster 3'

)



plt.scatter(

    X[y_km == 3, 0], X[y_km == 3, 1],

    s=150, c='yellow',

    label='cluster 4'

)









# plot the centroids

plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=15,

    c='red',label='centroids'

)





#plt.legend(scatterpoints=1)

plt.legend()

plt.title('Visualization of clustered data', fontweight='bold')

ax.set_aspect('equal');

plt.show()
model = TSNE(learning_rate=200)



# Apply fit_transform to samples: tsne_features

transformed = model.fit_transform(get_numeric_data)

transformed[1:1,:]
get_numeric_data.columns
get_numeric_data.shape
get_numeric_data['x'] = transformed[:,0]

get_numeric_data['y'] = transformed[:,1]
x = get_numeric_data.iloc[:, [3,4]].values #.iloc for positional indexing

y = get_numeric_data.iloc[:,].values
sns.scatterplot(get_numeric_data['x'] , get_numeric_data['y'] , data=get_numeric_data)



plt.title('After did TSNE', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
#set random seed

np.random.seed(200)
# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []

for cluster in range(1,20):

    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')

    kmeans.fit(transformed)

    SSE.append(kmeans.inertia_)



# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})

plt.figure(figsize=(12,6))

plt.plot(frame['Cluster'], frame['SSE'], marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')
k = 5  #ประการตัวแปร
x = get_numeric_data.iloc[:, [3,4]].values #.iloc for positional indexing

y = get_numeric_data.iloc[:,].values
# centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(-1, 80), np.random.randint(-1, 80)]

    for i in range(k)

}

    

fig = plt.figure(figsize=(10, 10))

plt.scatter(['x'],['y'], color='#5742FD')

colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'cyan', 5: 'yellow'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

    

plt.title('Visualization of clustered data', fontweight='bold')

#ax.set_aspect('equal');



plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
def assignment(df, centroids):

    for i in centroids.keys():

        # sqrt((x1 - x2)^2 - (y1 - y2)^2)

        df['distance_from_{}'.format(i)] = (

            np.sqrt(

                (get_numeric_data['x'] - centroids[i][0]) ** 2

                + (get_numeric_data['y'] - centroids[i][1]) ** 2

            )

        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df



#print(df.head())
df = assignment(get_numeric_data, centroids)

fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['x'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

    

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (get_numeric_data)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
old_centroids = copy.deepcopy(centroids)



def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])

        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])

    return k



centroids = update(centroids)

    

fig = plt.figure(figsize=(10,10))

ax = plt.axes()

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=0.5, head_length=0.5, fc=colmap[i], ec=colmap[i])



plt.title('K-Means (get_numeric_data)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
df = assignment(df, centroids)



# Plot results

fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['y'], color=df['color'],alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (get_numeric_data)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
# Continue until all assigned categories don't change any more

while True:

    closest_centroids = df['closest'].copy(deep=True)

    centroids = update(centroids)

    df = assignment(df, centroids)

    if closest_centroids.equals(df['closest']):

        break



fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (get_numeric_data)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()

get_numeric_data
new_data2 = get_numeric_data.drop(get_numeric_data.columns[[246]], axis=True).head(10)

new_data2
new_data2.replace([np.inf, - np.inf], np.nan)

new_data2.isnull().any()
new_data2.info()
X = new_data2.iloc[:, [3,4]].values #.iloc for positional indexing

y = new_data2.iloc[:,].values
km = KMeans(

    n_clusters=5, init='random',

    n_init=10, max_iter=300, 

    tol=1e-04

)



  



y_km = km.fit_predict(X)

plt.scatter(

    X[y_km == 0, 0], X[y_km == 0, 1],

    s=150, c='lightgreen',

    label='cluster 1'

)



plt.scatter(

    X[y_km == 1, 0], X[y_km == 1, 1],

    s=150, c='orange',

    label='cluster 2'

)



plt.scatter(

    X[y_km == 2, 0], X[y_km == 2, 1],

    s=150, c='lightblue',

    label='cluster 3'

)



plt.scatter(

    X[y_km == 3, 0], X[y_km == 3, 1],

    s=150, c='yellow',

    label='cluster 4'

)



plt.scatter(

    X[y_km == 4, 0], X[y_km == 4, 1],

    s=150, c='cyan',

    label='cluster 5'

)





# plot the centroids

plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=15,

    c='red',label='centroids'

)





#plt.legend(scatterpoints=1)

plt.legend()

plt.title('After Normalization data', fontweight='bold')

ax.set_aspect('equal');

plt.show()
x = new_data2.iloc[:, [3,4]].values #.iloc for positional indexing

y = new_data2.iloc[:,].values
k = 5



# centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(-1, 80), np.random.randint(-1, 80)]

    for i in range(k)

}

    

fig = plt.figure(figsize=(10, 10))

plt.scatter(['x'],['y'], color='#5742FD')

colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'cyan', 5: 'yellow'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

    

plt.title('Visualization of clustered data', fontweight='bold')

#ax.set_aspect('equal');



plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
def assignment(df, centroids):

    for i in centroids.keys():

        # sqrt((x1 - x2)^2 - (y1 - y2)^2)

        df['distance_from_{}'.format(i)] = (

            np.sqrt(

                (get_numeric_data['x'] - centroids[i][0]) ** 2

                + (get_numeric_data['y'] - centroids[i][1]) ** 2

            )

        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df



#print(df.head())
df = assignment(new_data2, centroids)



fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['x'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (new_data2)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['x'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (new_data2)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
old_centroids = copy.deepcopy(centroids)



def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])

        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])

    return k



centroids = update(centroids)

    

fig = plt.figure(figsize=(10,10))

ax = plt.axes()

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=0.5, head_length=0.5, fc=colmap[i], ec=colmap[i])

    

plt.title('K-Means (new_data2)', fontsize=18)    

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
df = assignment(df, centroids)



# Plot results

fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['y'], color=df['color'],alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (new_data2)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
# Continue until all assigned categories don't change any more

while True:

    closest_centroids = df['closest'].copy(deep=True)

    centroids = update(centroids)

    df = assignment(df, centroids)

    if closest_centroids.equals(df['closest']):

        break



fig = plt.figure(figsize=(10,10))

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

#plt.xlim(0, 80)

#plt.ylim(0, 80)

plt.title('K-Means (new_data2)', fontsize=18)

plt.xlabel('x', fontsize=18)

plt.ylabel('y', fontsize=18)

plt.show()
km = KMeans(

    n_clusters=5, init='random',

    n_init=10, max_iter=300, 

    tol=1e-04

)



  



y_km = km.fit_predict(x)
plt.scatter(

    x[y_km == 0, 0], x[y_km == 0, 1],

    s=150, c='lightgreen',

    label='cluster 1'

)



plt.scatter(

    x[y_km == 1, 0], x[y_km == 1, 1],

    s=150, c='orange',

    label='cluster 2'

)



plt.scatter(

    x[y_km == 2, 0], x[y_km == 2, 1],

    s=150, c='lightblue',

    label='cluster 3'

)



plt.scatter(

    X[y_km == 3, 0], X[y_km == 3, 1],

    s=150, c='yellow',

    label='cluster 4'

)



plt.scatter(

    x[y_km == 4, 0], x[y_km == 4, 1],

    s=150, c='cyan',

    label='cluster 5'

)





# plot the centroids

plt.scatter(

    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],

    s=15,

    c='red',label='centroids'

)





#plt.legend(scatterpoints=1)

plt.legend()

plt.title('After Normalization data', fontweight='bold')

ax.set_aspect('equal');

plt.show()