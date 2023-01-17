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
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
#df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
df.head()
# Import the required libraries
import os
import nltk


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import edit_distance

import re

import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np # linear algebra

import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
# directory = 'C:/Users/N1110/Desktop/CORD19/'
# df = pd.read_csv(directory + 'metadata.csv')
# df.head()
df.shape
df.dtypes
df.isnull().sum()
df1 = df[['doi','title','abstract']]
df2=df1.drop_duplicates(keep='last')
df2.isnull().sum()
df2.shape

df2.title.head(5)
type(df2.title.head(5)) # to list
df2.title.shape
df2.title.dropna(inplace=True)
df2.title.shape
df2.isnull().sum()
df2.dropna(inplace=True)
df2.shape
df2.isnull().sum()
# df3=df2.iloc[0:100,] #subset a small dataset for test

# df3
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

#df3['title'] = df3['title'].apply(lambda x: lower_case(x))
# df3.head()
# tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',max_features = 2**12)
# tf_idf = tf_idf_vectorizor.fit_transform(df3['title'])
# tf_idf_norm = normalize(tf_idf)
# tf_idf_array = tf_idf_norm.toarray()
# pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
# sklearn_pca = PCA(n_components = 10)
# X = sklearn_pca.fit_transform(tf_idf_array)
# kmeans = KMeans(n_clusters=10, max_iter=600, algorithm = 'auto')
# fitted = kmeans.fit(X)
# prediction = kmeans.predict(X)
# plt.scatter(X[:, 0], X[:, 1], c=prediction, s=50, cmap='viridis')
# centers = fitted.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
#df2['title'] = df2['title'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))
df2['abstract'] = df2['abstract'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]','',x))

#df2['title'] = df2['title'].apply(lambda x: lower_case(x))
df2['abstract'] = df2['abstract'].apply(lambda x: lower_case(x))
df2.drop(df2[df2['abstract']=='unknown'].index,inplace=True) #drop abstracts with content="unknown"
df2.shape  #check how many dropped  before (35177, 3) after (34989, 3)
df2.tail()
df2['abstract'].iloc[1] # i guess only full text need to remove /n
tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',max_features = 2**12)
tf_idf = tf_idf_vectorizor.fit_transform(df2.abstract)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()
pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names()).head()
sklearn_pca = PCA(n_components = 10)
X = sklearn_pca.fit_transform(tf_idf_array)
kmeans = KMeans(n_clusters=10, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(X)
prediction = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, s=50, cmap='viridis')
centers = fitted.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)
kmeans = KMeans(n_clusters=5, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(X)
prediction = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=prediction, s=50, cmap='viridis')
centers = fitted.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
#alternate

#run for quite long without pca  #should do title level first for this type of clustering. s first pca then clustering?

# X=pd.DataFrame(tf_idf_array, columns=tf_idf_vectorizor.get_feature_names())

sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)

cls = KMeans(n_clusters=3, init='k-means++',random_state=1) #n_clusters=3
cls.fit(X)
newfeature = cls.labels_ # the labels from kmeans clustering
X2 = np.column_stack((X,pd.get_dummies(newfeature)))
X2
#quick
plt.figure()
#plt.subplot(1,2,1)
X2=X2
plt.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow, s=20, linewidths=0,alpha=0.5)
plt.xlabel(''), plt.ylabel('')
plt.grid()
#3D plot

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


fig = pyplot.figure()
ax = Axes3D(fig)


ax.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2,X2[:, 2]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow,alpha=0.25)
pyplot.show()
#run for quite long
sse = []
list_k = list(range(1, 20))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, sse, '-o')
plt.xlabel(r'Number of clusters *k*')
plt.ylabel('Sum of squared distance');
#run for so long
data=X

# Silhouette vs Cluster Size
# do it for the k-means
from sklearn import metrics
from sklearn.cluster import KMeans

seuclid = []
scosine = []
k = range(2,11)
for i in k:
    kmeans_model = KMeans(n_clusters=i, init="k-means++").fit(X)
    labels = kmeans_model.labels_
    seuclid.append(metrics.silhouette_score(data, labels, metric='euclidean'))
    scosine.append(metrics.silhouette_score(data, labels, metric='cosine'))
    
plt.figure(figsize=(10,5))
plt.plot(k,seuclid,label='euclidean')
plt.plot(k,scosine,label='cosine')
plt.ylabel("Silhouette")
plt.xlabel("Cluster")
plt.title("Silhouette vs Cluster Size")
plt.legend()
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

print(__doc__)

# y_lower = 10?
X=X2
y=newfeature
range_n_clusters = [ 3, 4, 5, 6,7,8,9,10,11] # [3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] 


for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

#n_clusters=4

sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)

cls = KMeans(n_clusters=4, init='k-means++',random_state=1) 
cls.fit(X)
newfeature = cls.labels_ # the labels from kmeans clustering

X2 = np.column_stack((X,pd.get_dummies(newfeature)))


plt.figure()
#plt.subplot(1,2,1)
X2=X2
plt.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow, s=20, linewidths=0,alpha=0.5)
plt.xlabel(''), plt.ylabel('')
plt.grid()

#3D plot

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


fig = pyplot.figure()
ax = Axes3D(fig)


ax.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2,X2[:, 2]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow,alpha=0.25)
pyplot.show()
#n_clusters=10

sklearn_pca = PCA(n_components = 30)
X = sklearn_pca.fit_transform(tf_idf_array)

cls = KMeans(n_clusters=10, init='k-means++',random_state=1) 
cls.fit(X)
newfeature = cls.labels_ # the labels from kmeans clustering

X2 = np.column_stack((X,pd.get_dummies(newfeature)))


plt.figure()
#plt.subplot(1,2,1)
X2=X2
plt.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow, s=20, linewidths=0,alpha=0.5)
plt.xlabel(''), plt.ylabel('')
plt.grid()
#3D plot

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D


fig = pyplot.figure()
ax = Axes3D(fig)


ax.scatter(X2[:, 0], X2[:, 1]+np.random.random(X2[:, 1].shape)/2,X2[:, 2]+np.random.random(X2[:, 1].shape)/2, c=newfeature, cmap=plt.cm.rainbow,alpha=0.25)
pyplot.show()
