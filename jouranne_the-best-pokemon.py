import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/Pokemon.csv")

print(df.info())

print(df.shape)

df.head(12)

df.iloc[:,~df.columns.isin(['#','name','Type1','Type1','Legendary'])].describe()
corr=df.iloc[:,1:13].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(2, 110, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.6, cbar_kws={"shrink": .6})
#Quitando las columnas de clasificación y el Total (ya que no me interesa estudiar el comportamiento de esta bajo las otras habilidades) 

ds=df.drop(df.columns[[0,4,11,12]], axis=1)

sns.pairplot(ds)
df_scale = df.copy()

scaler = preprocessing.StandardScaler()

columns =df.columns[5:11]

df_scale[columns] = scaler.fit_transform(df_scale[columns])

df_scale.head(10)
#Elbow graph

ks = range(1, 10)

inertias = []



for k in ks:

    model = KMeans(n_clusters=k)

    model.fit(df_scale.iloc[:,5:])

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o')

plt.xlabel('Cantidad de clusters, k')

plt.ylabel('Inercia')

plt.xticks(ks)

plt.show()
model = KMeans(n_clusters=4)

model.fit(df_scale.iloc[:,5:10])

df_scale['cluster'] = model.predict(df_scale.iloc[:,5:10])

df_scale.head(20)
df_scale.iloc[:,~df_scale.columns.isin(['#','name','Type1','Type1','Legendary'])].groupby('cluster').mean()
# Replicando la grafica de la clase :)

model_pca = PCA()

pca_features = model_pca.fit_transform(df_scale.iloc[:,5:11])

xs = pca_features[:,0]

ys = pca_features[:,1]

sns.scatterplot(x=xs, y=ys, hue="cluster", data=df_scale)
Z = linkage(df_scale.iloc[:,5:11], 'complete')



plt.figure(figsize=(10, 150))

plt.title('Dendograma jerárquico')

plt.xlabel('Pokemon')

plt.ylabel('Distancia')

dg = df_scale.set_index('Name')





dendrogram(Z, labels=dg.index, leaf_rotation=0, orientation="left", leaf_font_size=12., show_contracted=False)

my_palette = plt.cm.get_cmap("tab10", 4)

df_scale['cluster']=pd.Categorical(df_scale['cluster'])

my_color=df_scale['cluster'].cat.codes

 

# Apply the right color to each label

ax = plt.gca()

xlbls = ax.get_ymajorticklabels()

num=-1

for lbl in xlbls:

    num+=1

    val=my_color[num]

    lbl.set_color(my_palette(val))

from scipy.cluster.hierarchy import fcluster

clusters2 = fcluster(Z, 9,criterion='distance')

df_scale['clusters2'] = clusters2



df_scale.iloc[:,~df_scale.columns.isin(['#','name','Type1','Type1','Legendary'])].groupby('clusters2').mean()

ax = sns.countplot(x="clusters2", hue="cluster", data=df_scale)
ax = sns.countplot(x="Legendary", hue="clusters2", data=df_scale)
df2 = df_scale[df_scale['clusters2'] == 2]



df2.head(10)

# Replicando la grafica de la clase :)

model_pca = PCA()

pca_features = model_pca.fit_transform(df_scale.iloc[:,5:11])

xs = pca_features[:,0]

ys = pca_features[:,1]

sns.scatterplot(x=xs, y=ys, hue="clusters2", data=df_scale)