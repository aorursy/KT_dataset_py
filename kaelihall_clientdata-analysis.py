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
import pandas as pd

ClientData = pd.read_csv("../input/client-data/ClientData.csv")

cdf = pd.read_csv("../input/client-data/ClientDataFlattened.csv")
import pandas as pd



cabdata = pd.read_csv("../input/cabdata/cab-data.csv")
cdf.head()
ati=cdf[[' Avg Ticket', ' Total Rev','Invoice Count']]

aptr=cdf[['Ad Profit',' Total Rev']] #Saving this one for later
from sklearn.cluster import KMeans 

clusters = 7

  

kmeans = KMeans(n_clusters = clusters) 

kmeans.fit(ati) #Inputting my variable

  

print(kmeans.labels_)
from sklearn.decomposition import PCA 

  

pca = PCA(3) 

pca.fit(ati) #Inputting my variable

  

pca_data = pd.DataFrame(pca.transform(ati)) 

  

print(pca_data.head())
from matplotlib import colors as mcolors 

import math 

   

''' Generating different colors in ascending order  

                                of their hsv values '''

colors = list(zip(*sorted(( 

                    tuple(mcolors.rgb_to_hsv( 

                          mcolors.to_rgba(color)[:3])), name) 

                     for name, color in dict( 

                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS 

                                                      ).items())))[1] 

   

   

# number of steps to taken generate n(clusters) colors  

skips = math.floor(len(colors[5 : -5])/clusters) 

cluster_colors = colors[5 : -5 : skips] 
from mpl_toolkits.mplot3d import Axes3D 

import matplotlib.pyplot as plt 

   

fig = plt.figure() 

ax = fig.add_subplot(111, projection = '3d') 

ax.scatter(pca_data[0], pca_data[1], pca_data[2],  

           c = list(map(lambda label : cluster_colors[label], 

                                            kmeans.labels_))) 

   

str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 

   

list(map(lambda data1, data2, data3, str_label: 

        ax.text(data1, data2, data3, s = str_label, size = 0, 

        zorder = 20, color = 'k'), pca_data[0], pca_data[1], 

        pca_data[2], str_labels)) 

   

plt.show() 
import seaborn as sns 

  

# generating correlation heatmap 

sns.heatmap(cdf.corr(), annot = True) 

  

# posting correlation heatmap to output console  

plt.show() 
atic=cdf[['Invoice Count',' Avg Ticket']] #

atic.head()
from matplotlib import cm 

  

# generating correlation data 

df = cdf.corr() 

df.index = range(0, len(df)) 

df.rename(columns = dict(zip(df.columns, df.index)), inplace = True) 

df = df.astype(object) 

  

''' Generating coordinates with  

corresponding correlation values '''

for i in range(0, len(df)): 

    for j in range(0, len(df)): 

        if i != j: 

            df.iloc[i, j] = (i, j, df.iloc[i, j]) 

        else : 

            df.iloc[i, j] = (i, j, 0) 

  

df_list = [] 

  

# flattening dataframe values 

for sub_list in df.values: 

    df_list.extend(sub_list) 

  

# converting list of tuples into trivariate dataframe 

plot_df = pd.DataFrame(df_list) 

  

fig = plt.figure() 

ax = Axes3D(fig) 

  

# plotting 3D trisurface plot 

ax.plot_trisurf(plot_df[0], plot_df[1], plot_df[2],  

                    cmap = cm.jet, linewidth = 0.2) 

  

plt.show() 
cabaptr=cabdata[['Ad Profit',' Total Rev']]

cabaptr.head(20)

#cabaptr=cabaptr.drop([4, 9])
cab = [

    [49839.93,  1192367.01],

    [15301.71,  944999.18],

    [21037.05,  3472122.95],

    [84422.88,  626103.22],

    [7556.43,   1906449.30],

    [10680.37,  6437677.03],

    [6467.97,   2528795.20],

    [43137.96,  1763236.44],

    [16008.06,  1036380.79],

    [25763.58,  2463164.03],

    [8525.98,   624811.99],

    [26605.45,  1402229.65],

    [24212.52,  2792226.84],

    [2587.51,   1477251.33],

    [1830.71,   1948668.57]

]

cb = pd.DataFrame(cab)


import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



df = aptr



np.random.seed(200)

k = 3

# centroids[i] = [x, y]

centroids = {

    i+1: [np.random.randint(0, 50000), np.random.randint(0, 8000000)]

    for i in range(k)

}

    

fig = plt.figure(figsize=(5, 5))

plt.scatter(df['Ad Profit'], df[' Total Rev'], color='k')

#plt.scatter(cabaptr['Ad Profit'], cabaptr[' Total Rev'], color='yellow')

colmap = {1: 'r', 2: 'g', 3: 'b'}

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 50000)

plt.ylim(0, 8000000)

plt.show()
plt.scatter(x=cb[0], y=cb[1], color='yellow')#Plotting my Advisory board data to compare

plt.xlim(0, 90000)

plt.ylim(0, 8000000)
def assignment(df, centroids):

    for i in centroids.keys():

        # sqrt((x1 - x2)^2 - (y1 - y2)^2)

        df['distance_from_{}'.format(i)] = (

            np.sqrt(

                (df['Ad Profit'] - centroids[i][0]) ** 2

                + (df[' Total Rev'] - centroids[i][1]) ** 2

            )

        )

    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]

    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)

    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))

    df['color'] = df['closest'].map(lambda x: colmap[x])

    return df



df = assignment(df, centroids)

print(df.head())



fig = plt.figure(figsize=(5, 5))

plt.scatter(df['Ad Profit'], df[' Total Rev'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 50000)

plt.ylim(0, 8000000)

plt.show()


import copy



old_centroids = copy.deepcopy(centroids)



def update(k):

    for i in centroids.keys():

        centroids[i][0] = np.mean(df[df['closest'] == i]['Ad Profit'])

        centroids[i][1] = np.mean(df[df['closest'] == i][' Total Rev'])

    return k



centroids = update(centroids)

    

fig = plt.figure(figsize=(5, 5))

ax = plt.axes()

plt.scatter(df['Ad Profit'], df[' Total Rev'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 50000)

plt.ylim(0, 8000000)

for i in old_centroids.keys():

    old_x = old_centroids[i][0]

    old_y = old_centroids[i][1]

    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75

    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75

    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])

plt.show()
## Repeat Assigment Stage



df = assignment(df, centroids)



# Plot results

fig = plt.figure(figsize=(5, 5))

plt.scatter(df['Ad Profit'], df[' Total Rev'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 50000)

plt.ylim(0, 8000000)

plt.show()
while True:

    closest_centroids = df['closest'].copy(deep=True)

    centroids = update(centroids)

    df = assignment(df, centroids)

    if closest_centroids.equals(df['closest']):

        break



fig = plt.figure(figsize=(5, 5))

plt.scatter(df['Ad Profit'], df[' Total Rev'], color=df['color'], alpha=0.5, edgecolor='k')

plt.scatter(x=cb[0], y=cb[1], color='yellow')

for i in centroids.keys():

    plt.scatter(*centroids[i], color=colmap[i])

plt.xlim(0, 50000)

plt.ylim(0, 8000000)

plt.xlabel('Ad Profit')

plt.ylabel('Total Revenue')

plt.show()
cabatic=cabdata[[' Avg Ticket', ' Car Count']]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(atic)

aticnormal = scaler.transform(atic)#Normalizing my entire database

cabnormal = scaler.transform(cabatic)#Using the same scaler to normalize my Advisory Board data

aticnormal
cabnormal
#The normalizer changed my data into arrays, so I have to put them back into dataframes



aticnormaldf=pd.DataFrame(aticnormal)

cabnormaldf=pd.DataFrame(cabnormal)
cabnormaldf.head(30)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(aticnormaldf)
labels = kmeans.predict(aticnormaldf)

centroids = kmeans.cluster_centers_
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

kmeans.fit(aticnormaldf)
labels = kmeans.predict(aticnormaldf)

centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(10, 10))



colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(aticnormaldf[0], aticnormaldf[1], color=colors, alpha=0.5, edgecolor='k')

plt.scatter(cabnormaldf[0], cabnormaldf[1], color='yellow')

for idx, centroid in enumerate(centroids):

    plt.scatter(*centroid, color=colmap[idx+1])

plt.xlim(0,1)

plt.ylim(0, 1)

plt.xlabel('Average Ticket')

plt.ylabel('Invoice Count')

plt.show()
cdf.head()
atnc=cdf[[" Avg Ticket", ' New Customers']]

cabatnc=cabdata[[' Avg Ticket', " New Customers"]]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(atnc)

atncnormal = scaler.transform(atnc)#Normalizing my entire database

cabatncnormal = scaler.transform(cabatnc)#Using the same scaler to normalize my Advisory Board data

atncnormaldf=pd.DataFrame(atncnormal)

cabatncnormaldf=pd.DataFrame(cabatncnormal)
kmeans = KMeans(n_clusters=3)

kmeans.fit(atncnormaldf)
labels = kmeans.predict(atncnormaldf)

centroids = kmeans.cluster_centers_
fig = plt.figure(figsize=(10, 10))



colors = list(map(lambda x: colmap[x+1], labels))

plt.scatter(atncnormaldf[0], atncnormaldf[1], color=colors, alpha=0.5, edgecolor='k')

plt.scatter(cabatncnormaldf[0], cabatncnormaldf[1], color='yellow')

for idx, centroid in enumerate(centroids):

    plt.scatter(*centroid, color=colmap[idx+1])

plt.xlim(0,1)

plt.ylim(0, 1)

plt.xlabel('Average Ticket')

plt.ylabel('New Customer')

plt.show()