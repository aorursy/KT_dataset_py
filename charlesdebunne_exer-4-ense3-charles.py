# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt

from sklearn import metrics

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
metrics.homogeneity_score([0, 0, 1, 1], [1, 1, 0, 0])
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 0, 1, 2]))
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 1, 2, 3]))
print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1]))

print("%.3f" % metrics.homogeneity_score([0, 0, 1, 1], [0, 0, 0, 0]))

print (metrics.completeness_score([0, 0, 1, 1], [1, 1, 0, 0]))

print(metrics.completeness_score([0, 0, 1, 1], [0, 0, 0, 0]))

print(metrics.completeness_score([0, 1, 2, 3], [0, 0, 1, 1]))

print(metrics.completeness_score([0, 0, 1, 1], [0, 1, 0, 1]))

print(metrics.completeness_score([0, 0, 0, 0], [0, 1, 2, 3]))
print (metrics.v_measure_score([0, 0, 1, 1], [0, 0, 1, 1]))

print (metrics.v_measure_score([0, 0, 1, 1], [1, 1, 0, 0]))
print("%.3f" % metrics.completeness_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.homogeneity_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.v_measure_score([0, 1, 2, 3], [0, 0, 0, 0]))

print("%.3f" % metrics.v_measure_score([0, 0, 1, 2], [0, 0, 1, 1]))

print("%.3f" % metrics.v_measure_score([0, 1, 2, 3], [0, 0, 1, 1]))
print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 0, 1, 2]))

print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 1, 2, 3]))
print("%.3f" % metrics.v_measure_score([0, 0, 0, 0], [0, 1, 2, 3]))
print("%.3f" % metrics.v_measure_score([0, 0, 1, 1], [0, 0, 0, 0]))
#Create some data

MAXN=40

X = np.concatenate([1.25*np.random.randn(MAXN,2), 5+1.5*np.random.randn(MAXN,2)])

X = np.concatenate([X,[8,3]+1.2*np.random.randn(MAXN,2)])

X.shape
#Just for visualization purposes, create the labels of the 3 distributions

y = np.concatenate([np.ones((MAXN,1)),2*np.ones((MAXN,1))])

y = np.concatenate([y,3*np.ones((MAXN,1))])



plt.subplot(1,2,1)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')

plt.title('Data as were generated')



plt.subplot(1,2,2)

plt.scatter(X[:,0],X[:,1],color='r')

plt.title('Data as the algorithm sees them')



plt.savefig("/kaggle/input/guithub-import-ense3/files/ch07/sample.png",dpi=300, bbox_inches='tight')



from sklearn import cluster



K=3 # Assuming to be 3 clusters!



clf = cluster.KMeans(init='random', n_clusters=K)

clf.fit(X)
print (clf.labels_) # or

print (clf.predict(X)) # equivalent
print (X[(y==1).ravel(),0]) #numpy.ravel() returns a flattened array

print (X[(y==1).ravel(),1])
plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')



fig = plt.gcf()

fig.set_size_inches((6,5))
x = np.linspace(-5,15,200)

XX,YY = np.meshgrid(x,x)

sz=XX.shape

data=np.c_[XX.ravel(),YY.ravel()]

# c_ translates slice objects to concatenation along the second axis.
Z=clf.predict(data) # returns the labels of the data

print (Z)
# Visualize space partition

plt.imshow(Z.reshape(sz), interpolation='bilinear', origin='lower',

extent=(-5,15,-5,15),alpha=0.3, vmin=0, vmax=K-1)

plt.title('Space partitions', size=14)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')



fig = plt.gcf()

fig.set_size_inches((6,5))



plt.savefig("files/ch07/samples3.png",dpi=300, bbox_inches='tight')
clf = cluster.KMeans(n_clusters=K, random_state=0)

#initialize the k-means clustering

clf.fit(X) #run the k-means clustering



data=np.c_[XX.ravel(),YY.ravel()]

Z=clf.predict(data) # returns the clustering labels of the data
plt.title('Final result of K-means', size=14)



plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b')

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g')



plt.imshow(Z.reshape(sz), interpolation='bilinear', origin='lower',

extent=(-5,15,-5,15),alpha=0.3, vmin=0, vmax=K-1)



x = np.linspace(-5,15,200)

XX,YY = np.meshgrid(x,x)

fig = plt.gcf()

fig.set_size_inches((6,5))



plt.savefig("files/ch07/randscore.png",dpi=300, bbox_inches='tight')
clf = cluster.KMeans(init='random', n_clusters=K, random_state=0)

#initialize the k-means clustering

clf.fit(X) #run the k-means clustering

Zx=clf.predict(X)



plt.subplot(1,3,1)

plt.title('Original labels', size=14)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='b') # b

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='g') # g

fig = plt.gcf()

fig.set_size_inches((12,3))



plt.subplot(1,3,2)

plt.title('Data without labels', size=14)

plt.scatter(X[(y==1).ravel(),0],X[(y==1).ravel(),1],color='r')

plt.scatter(X[(y==2).ravel(),0],X[(y==2).ravel(),1],color='r') # b

plt.scatter(X[(y==3).ravel(),0],X[(y==3).ravel(),1],color='r') # g

fig = plt.gcf()

fig.set_size_inches((12,3))



plt.subplot(1,3,3)

plt.title('Clustering labels', size=14)

plt.scatter(X[(Zx==1).ravel(),0],X[(Zx==1).ravel(),1],color='r')

plt.scatter(X[(Zx==2).ravel(),0],X[(Zx==2).ravel(),1],color='b')

plt.scatter(X[(Zx==0).ravel(),0],X[(Zx==0).ravel(),1],color='g')

fig = plt.gcf()

fig.set_size_inches((12,3))
from sklearn import metrics



clf = cluster.KMeans(n_clusters=K, init='k-means++', random_state=0,

max_iter=300, n_init=10)

#initialize the k-means clustering

clf.fit(X) #run the k-means clustering



print ('Final evaluation of the clustering:')



print('Inertia: %.2f' % clf.inertia_)



print('Adjusted_rand_score %.2f' % metrics.adjusted_rand_score(y.ravel(),

clf.labels_))



print('Homogeneity %.2f' % metrics.homogeneity_score(y.ravel(),

clf.labels_))



print('Completeness %.2f' % metrics.completeness_score(y.ravel(),

clf.labels_))



print('V_measure %.2f' % metrics.v_measure_score(y.ravel(), clf.labels_))



print('Silhouette %.2f' % metrics.silhouette_score(X, clf.labels_,

metric='euclidean'))



clf1 = cluster.KMeans(n_clusters=K, init='random', random_state=0,

max_iter=2, n_init=2)

#initialize the k-means clustering

clf1.fit(X) #run the k-means clustering



print ('Final evaluation of the clustering:')



print ('Inertia: %.2f' % clf1.inertia_)



print ('Adjusted_rand_score %.2f' % metrics.adjusted_rand_score(y.ravel(),

clf1.labels_))



print ('Homogeneity %.2f' % metrics.homogeneity_score(y.ravel(),

clf1.labels_))



print ('Completeness %.2f' % metrics.completeness_score(y.ravel(),

clf1.labels_))



print ('V_measure %.2f' % metrics.v_measure_score(y.ravel(),

clf1.labels_))



print ('Silhouette %.2f' % metrics.silhouette_score(X, clf1.labels_,

metric='euclidean'))
from sklearn.preprocessing import StandardScaler

from sklearn import cluster



edu=pd.read_csv('./files/ch07/educ_figdp_1_Data.csv',na_values=':')

edu.head()
edu.tail()
#Pivot table in order to get a nice feature vector representation with dual indexing by TIME and GEO

pivedu=pd.pivot_table(edu, values='Value', index=['TIME', 'GEO'], columns=['INDIC_ED'])

pivedu.head()
print ('Let us check the two indices:\n')

print ('\nPrimary index (TIME): \n' + str(pivedu.index.levels[0].tolist()))

print ('\nSecondary index (GEO): \n' + str(pivedu.index.levels[1].tolist()))
#Extract 2010 set of values

edu2010=pivedu.ix[2010]

edu2010.head()
#Store column names and clear them for better handling. Do the same with countries

edu2010 = edu2010.rename(index={'Euro area (13 countries)': 'EU13',

'Euro area (15 countries)': 'EU15',

'European Union (25 countries)': 'EU25',

'European Union (27 countries)': 'EU27',

'Former Yugoslav Republic of Macedonia, the': 'Macedonia',

'Germany (until 1990 former territory of the FRG)': 'Germany'

})

features = edu2010.columns.tolist()



countries = edu2010.index.tolist()



edu2010.columns=range(12)

edu2010.head()
#Check what is going on in the NaN data

nan_countries=np.sum(np.where(edu2010.isnull(),1,0),axis=1)

plt.bar(np.arange(nan_countries.shape[0]),nan_countries)

plt.xticks(np.arange(nan_countries.shape[0]),countries,rotation=90,horizontalalignment='left',

fontsize=12)

fig = plt.gcf()

fig.set_size_inches((12,5))
#Remove non info countries

wrk_countries = nan_countries<4



educlean=edu2010.ix[wrk_countries] #.ix - Construct an open mesh from multiple sequences.



#Let us check the features we have

na_features = np.sum(np.where(educlean.isnull(),1,0),axis=0)

print (na_features)



plt.bar(np.arange(na_features.shape[0]),na_features)

plt.xticks(fontsize=12)

fig = plt.gcf()

fig.set_size_inches((8,4))
#Option A fills those features with some value, at risk of extracting wrong information

#Constant filling : edufill0=educlean.fillna(0)

edufill=educlean.fillna(educlean.mean())

print ('Filled in data shape: ' + str(edufill.shape))



#Option B drops those features

edudrop=educlean.dropna(axis=1)

#dropna: Return object with labels on given axis omitted where alternately any or

# all of the data are missing

print ('Drop data shape: ' + str(edudrop.shape))
scaler = StandardScaler() #Standardize features by removing the mean and scaling to unit variance



X_train_fill = edufill.values

X_train_fill = scaler.fit_transform(X_train_fill)



clf = cluster.KMeans(init='k-means++', n_clusters=3, random_state=42)



clf.fit(X_train_fill) #Compute k-means clustering.



y_pred_fill = clf.predict(X_train_fill)

#Predict the closest cluster each sample in X belongs to.



idx=y_pred_fill.argsort()
plt.plot(np.arange(35),y_pred_fill[idx],'ro')

wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]



plt.xticks(np.arange(len(wrk_countries_names)),[wrk_countries_names[i] for i in idx],

rotation=90,horizontalalignment='left',fontsize=12)

plt.title('Using filled in data', size=15)

plt.yticks([0,1,2])

fig = plt.gcf()



fig.set_size_inches((12,5))
X_train_drop = edudrop.values

X_train_drop = scaler.fit_transform(X_train_drop)



clf.fit(X_train_drop) #Compute k-means clustering.

y_pred_drop = clf.predict(X_train_drop) #Predict the closest cluster of each sample in X.
idx=y_pred_drop.argsort()

plt.plot(np.arange(35),y_pred_drop[idx],'ro')

wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]



plt.xticks(np.arange(len(wrk_countries_names)),[wrk_countries_names[i] for i in idx],

rotation=90,horizontalalignment='left',fontsize=12)

plt.title('Using dropped missing values data',size=15)

fig = plt.gcf()

plt.yticks([0,1,2])

fig.set_size_inches((12,5))
plt.plot(y_pred_drop+0.2*np.random.rand(35),y_pred_fill+0.2*np.random.rand(35),'bo')

plt.xlabel('Predicted clusters for the filled in dataset.')

plt.ylabel('Predicted clusters for the dropped missing values dataset.')

plt.title('Correlations')

plt.xticks([0,1,2])

plt.yticks([0,1,2])

plt.savefig("files/ch07/correlationkmeans.png",dpi=300, bbox_inches='tight')
print ('Cluster 0: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_fill)

if item==0]))

print ('Cluster 0: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_drop)

if item==0]))

print ('\n')

print ('Cluster 1: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_fill)

if item==1]))

print ('Cluster 1: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_drop)

if item==1]))

print ('\n')

print ('Cluster 2: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_fill)

if item==2]))

print ('Cluster 2: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred_drop)

if item==2]))

print ('\n')
width=0.3

p1 = plt.bar(np.arange(8),scaler.inverse_transform(clf.cluster_centers_[1]),width,color='b')

# Scale back the data to the original representation

p2 = plt.bar(np.arange(8)+width,scaler.inverse_transform(clf.cluster_centers_[2]),

width,color='yellow')

p0 = plt.bar(np.arange(8)+2*width,scaler.inverse_transform(clf.cluster_centers_[0]),

width,color='r')



plt.legend( (p0[0], p1[0], p2[0]), ('Cluster 0', 'Cluster 1', 'Cluster 2') ,loc=9)

plt.xticks(np.arange(8) + 0.5, np.arange(8),size=12)

plt.yticks(size=12)

plt.xlabel('Economical indicators')

plt.ylabel('Average expanditure')

fig = plt.gcf()



plt.savefig("files/ch07/clusterexpenditure.png",dpi=300, bbox_inches='tight')
from scipy.spatial import distance

p = distance.cdist(X_train_drop[y_pred_drop==0,:],[clf.cluster_centers_[1]],'euclidean')

#the distance of the elements of cluster 0 to the center of cluster 1



fx = np.vectorize(np.int)



plt.plot(np.arange(p.shape[0]),

fx(p)

)



wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]

zero_countries_names = [wrk_countries_names[i] for i,item in enumerate(y_pred_drop)

if item==0]

plt.xticks(np.arange(len(zero_countries_names)),zero_countries_names,rotation=90,

horizontalalignment='left',fontsize=12)
from scipy.spatial import distance

p = distance.cdist(X_train_drop[y_pred_drop==0,:],[clf.cluster_centers_[1]],'euclidean')

pown = distance.cdist(X_train_drop[y_pred_drop==0,:],[clf.cluster_centers_[0]],'euclidean')



width=0.45

p0=plt.plot(np.arange(p.shape[0]),fx(p),width)

p1=plt.plot(np.arange(p.shape[0])+width,fx(pown),width,color = 'red')



wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]

zero_countries_names = [wrk_countries_names[i] for i,item in enumerate(y_pred_drop)

if item==0]

plt.xticks(np.arange(len(zero_countries_names)),zero_countries_names,rotation=90,

horizontalalignment='left',fontsize=12)

plt.legend( (p0[0], p1[0]), ('d -> 1', 'd -> 0') ,loc=1)

plt.savefig("files/ch07/dist2cluster01.png",dpi=300, bbox_inches='tight')
X_train = edudrop.values

clf = cluster.KMeans(init='k-means++', n_clusters=4, random_state=0)

clf.fit(X_train)

y_pred = clf.predict(X_train)



idx=y_pred.argsort()

plt.plot(np.arange(35),y_pred[idx],'ro')

wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]



plt.xticks(np.arange(len(wrk_countries_names)),[wrk_countries_names[i] for i in idx],rotation=90,

horizontalalignment='left',fontsize=12)

plt.title('Using drop features',size=15)

plt.yticks([0,1,2,3])

fig = plt.gcf()

fig.set_size_inches((12,5))
width=0.2

p0 = plt.bar(np.arange(8)+1*width,clf.cluster_centers_[0],width,color='r')

p1 = plt.bar(np.arange(8),clf.cluster_centers_[1],width,color='b')

p2 = plt.bar(np.arange(8)+3*width,clf.cluster_centers_[2],width,color='yellow')

p3 = plt.bar(np.arange(8)+2*width,clf.cluster_centers_[3],width,color='pink')



plt.legend( (p0[0], p1[0], p2[0], p3[0]), ('Cluster 0', 'Cluster 1', 'Cluster 2',

'Cluster 3') ,loc=9)

plt.xticks(np.arange(8) + 0.5, np.arange(8),size=12)

plt.yticks(size=12)

plt.xlabel('Economical indicator')

plt.ylabel('Average expenditure')

fig = plt.gcf()

fig.set_size_inches((12,5))

plt.savefig("files/ch07/distances4clusters.png",dpi=300, bbox_inches='tight')
print ('Cluster 0: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred) if item==0]))



print ('Cluster 1: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred) if item==1]))



print ('Cluster 2: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred) if item==2]))



print ('Cluster 3: \n' + str([wrk_countries_names[i] for i,item in enumerate(y_pred) if item==3]))



#Save data for future use.

import pickle

ofname = open('edu2010.pkl', 'wb')

s = pickle.dump([edu2010, wrk_countries_names,y_pred ],ofname)

ofname.close()
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.spatial.distance import pdist

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import kneighbors_graph

from sklearn.metrics import euclidean_distances



X = StandardScaler().fit_transform(edudrop.values)



distances = euclidean_distances(edudrop.values)



spectral = cluster.SpectralClustering(n_clusters=4, affinity="nearest_neighbors")

spectral.fit(edudrop.values)



y_pred = spectral.labels_.astype(np.int)
idx=y_pred.argsort()



plt.plot(np.arange(35),y_pred[idx],'ro')

wrk_countries_names = [countries[i] for i,item in enumerate(wrk_countries) if item ]



plt.xticks(np.arange(len(wrk_countries_names)),[wrk_countries_names[i]

for i in idx],rotation=90,horizontalalignment='left',fontsize=12)



plt.yticks([0,1,2,3])



plt.title('Applying Spectral Clustering on the drop features',size=15)

fig = plt.gcf()

fig.set_size_inches((12,5))
X_train = edudrop.values

dist = pdist(X_train,'euclidean')

linkage_matrix = linkage(dist,method = 'complete');

plt.figure() # we need a tall figure

fig = plt.gcf()

fig.set_size_inches((12,12))

dendrogram(linkage_matrix, orientation="right", color_threshold = 4,labels = wrk_countries_names, leaf_font_size=20);



plt.savefig("files/ch07/ACCountires.png",dpi=300, bbox_inches='tight')

plt.show()



#plt.tight_layout() # fixes margins