import pandas as pd

import sqlite3



con = sqlite3.connect("../input/database.sqlite")

team=pd.read_sql_query('select * from Team',con)

team_attr=pd.read_sql_query('select * from Team_Attributes',con)

con.close()



df = pd.merge(team, team_attr, how='inner', left_on='team_api_id', right_on='team_api_id')

print (df.shape)
cols_to_keep = ['date', 'team_long_name',   u'buildUpPlaySpeed', u'buildUpPlayDribbling',

         u'buildUpPlayPassing', u'chanceCreationPassing', u'chanceCreationCrossing',

       u'chanceCreationShooting', u'defencePressure', u'defenceAggression', u'defenceTeamWidth']

df = df[cols_to_keep]
aggs = df.groupby('team_long_name')['date'].max().to_frame()

df.drop('date', axis=1, inplace=True)

df.drop_duplicates(subset='team_long_name', keep='last', inplace=True)

df = df.merge(right=aggs, right_index=True, left_on='team_long_name', how='right')

df = df.dropna()

df.set_index('team_long_name', inplace=True)

df.drop('date', axis=1, inplace=True)

print (df.shape)
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



x = df.copy()

x_normalized = StandardScaler().fit(x).transform(x)  



pca = PCA(n_components = 3).fit(x_normalized)

print (pca.explained_variance_ratio_)
from sklearn.manifold import MDS



mds = MDS(n_components = 2, n_init = 10)

mds_2 = MDS(n_components = 3, n_init = 10)

x_mds = mds.fit_transform(x_normalized)

x_mds_2 = mds_2.fit_transform(x_normalized)
team_names = ['Liverpool', 'Manchester United', 'Arsenal', 'Chelsea']

team_maps = {}

for team in team_names:

    print (team, df.index.get_loc(team))

    team_maps[team] = df.index.get_loc(team)
def diff(a,b):

    sum = 0

    for i in range(len(a)):

        sum += (a[i]-b[i])*(a[i]-b[i])

    return sum
print ("Before MDS: ")

i=-1

for team in team_names:

    i+=1

    for j in range(i+1,4):

        print ("{} and {} diff : {}".format(team, team_names[j],

                                            diff(df.iloc[team_maps[team]].tolist(), 

                                            df.iloc[team_maps[team_names[j]]].tolist())))
print ("After MDS (keeping 2 components) : ")

i=-1

for team in team_names:

    i+=1

    for j in range(i+1,4):

        print ("{} and {} diff : {}".format(team, team_names[j], 

                                           diff(x_mds[team_maps[team]], x_mds[team_maps[team_names[j]]])))
print ("After MDS (keeping 3 components) : ")

i=-1

for team in team_names:

    i+=1

    for j in range(i+1,4):

        print ("{} and {} diff : {}".format(team, team_names[j], 

                                           diff(x_mds_2[team_maps[team]], x_mds_2[team_maps[team_names[j]]])))
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

plt.rcParams['figure.figsize'] = (30,30)

def plot_labelled_scatter(X, y, class_labels, team_maps):

    num_labels = len(class_labels)



    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1



    marker_array = ['o', '^', '*']

    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA']

    cmap_bold = ListedColormap(color_array)

    bnorm = BoundaryNorm(numpy.arange(0, num_labels + 1, 1), ncolors=num_labels)

    plt.figure()



    plt.scatter(X[:, 0], X[:, 1], s=65, c=y, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)

    #ids_names = zip(ids, team_names)

    for i, x, y in zip(range(len(X[:,0])), X[:, 0], X[:, 1]):       

        if i in team_maps.values():

            for team in team_maps.items():

                if team[1] == i:

                    team_name = team[0]

                

            plt.annotate(

                team_name,

                #next(x[1] for x in ids_names if x[0]==i),

                xy=(x, y), xytext=(-20, 20),

                textcoords='offset points', ha='right', va='bottom',

                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))



    plt.xlim(x_min, x_max)

    plt.ylim(y_min, y_max)



    h = []

    for c in range(0, num_labels):

        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))

    plt.legend(handles=h)



    plt.show()





from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler



kmeans = KMeans(n_clusters = 3, random_state=10)

kmeans.fit(x_mds)

#ids = zip(ids, team_names)

#print([id[0] for id in ids])

#data_labels = df['team_long_name'].tolist()

plot_labelled_scatter(x_mds, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'], team_maps)
from matplotlib.colors import ListedColormap, BoundaryNorm

import numpy

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from mpl_toolkits.mplot3d import Axes3D



plt.rcParams['figure.figsize'] = (30,30)

def plot_labelled_scatter_3D(X, y, class_labels, team_maps):

    num_labels = len(class_labels)



    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    marker_array = ['o', '^', '*']

    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA']

    cmap_bold = ListedColormap(color_array)

    bnorm = BoundaryNorm(numpy.arange(0, num_labels + 1, 1), ncolors=num_labels)

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')



    ax.scatter(X[:, 0], X[:, 1], X[:,2], s=65, c=y, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)

    #ids_names = zip(ids, team_names)

    for i, x, y, z in zip(range(len(X[:,0])), X[:, 0], X[:, 1], X[:, 2]):       

        if i in team_maps.values():

            for team in team_maps.items():

                if team[1] == i:

                    team_name = team[0]

                

            ax.text(x, y, z, team_name)

            #ax.annotate(

            #    team_name,

                #next(x[1] for x in ids_names if x[0]==i),

            #    xy=(x, y), xytext=(-20, 20),

            #    textcoords='offset points', ha='right', va='bottom',

            #    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),

            #    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))



    plt.xlim(x_min, x_max)

    plt.ylim(y_min, y_max)

    #plt.zlim(z_min, z_max)

    ax.set_zlim(z_min, z_max)

    

    h = []

    for c in range(0, num_labels):

        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))

    plt.legend(handles=h)



    plt.show()





from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler



kmeans = KMeans(n_clusters = 3, random_state=10)

kmeans.fit(x_mds)

#ids = zip(ids, team_names)

#print([id[0] for id in ids])

#data_labels = df['team_long_name'].tolist()

plot_labelled_scatter_3D(x_mds_2, kmeans.labels_, ['Cluster 1', 'Cluster 2', 'Cluster 3'], team_maps)