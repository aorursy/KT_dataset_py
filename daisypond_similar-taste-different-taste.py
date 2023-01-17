!pip install pydotplus



import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from IPython.display import Image, display_png

from pydotplus import graph_from_dot_data

from sklearn.tree import export_graphviz

from sklearn import manifold

import folium

from pyproj import Proj, transform
df = pd.read_csv('../input/whisky.csv')

df.head()
df.info()
df.describe()
dist = []



for i in range(2,20):

    km = KMeans(n_clusters = i, n_init=10, max_iter = 500, random_state =0)

    km.fit(df.iloc[:, 2:-3])

    dist.append(km.inertia_)

    

plt.plot(range(2,20),dist)

plt.show()
km = KMeans(n_clusters = 5, n_init=10, max_iter = 300, random_state =0)

df['class'] = km.fit_predict(df.iloc[:, 2:-3])

df['class'].values
mds = manifold.MDS(n_components=2, dissimilarity="euclidean", random_state=0)

pos = mds.fit_transform(df.iloc[:, 2:-4])



col =['orange','green', 'blue', 'purple', 'red']

chars = "^<>vo+d"

c_flag = 0

labels = df['Distillery']



plt.figure(figsize=(20, 20), dpi=50)

plt.rcParams["font.size"] = 15



for label, x, y, c in zip(labels, pos[:, 0], pos[:, 1],df['class']):



    if(c == c_flag):

        c_flag = c_flag+1

        plt.scatter(x,y, c=col[c], marker=chars[c], s=100, label = "Class "+ str(c+1))

    else:

        plt.scatter(x,y, c=col[c], marker=chars[c], s=100)

        

    plt.annotate(label,xy = (x, y))

plt.legend(loc='upper right')

plt.show()
df.query('Distillery == "GlenSpey" or Distillery == "Miltonduff"')
df.query('Distillery == "GlenSpey" or Distillery == "Glendronach"')
tree = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state =1, min_samples_leaf=5)



X_train = df.iloc[:, 2:-4]

y_train = df['class']



tree.fit(X_train, y_train)
dot_data = export_graphviz(tree, filled = True, rounded = True, class_names = ['Class 1','Class 2', 'Class 3', 'Class 4', 'Class 5'],

                          feature_names = df.columns[2:-4].values, out_file = None)



graph = graph_from_dot_data(dot_data)

graph.write_png('tree.png')

display_png(Image('tree.png'))
map_whisky = folium.Map(location=[57.499520,  -2.776390], zoom_start = 9)



inProj = Proj(init='epsg:27700')

outProj = Proj(init='epsg:4326')



for label, lon, lat, c in zip(labels, df['Latitude'], df['Longitude'], df['class']):

    

    lat2,lon2 = transform(inProj,outProj,lon,lat)

    folium.Marker([lon2, lat2], popup= label, icon=folium.Icon(color=col[c])).add_to(map_whisky)



map_whisky