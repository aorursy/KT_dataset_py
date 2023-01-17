

# Supress Warnings

import warnings

warnings.filterwarnings('ignore')

# Importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.cm as cm

# visulaisation

from matplotlib.pyplot import xticks

%matplotlib inline

# Data display coustomization

pd.set_option('display.max_columns', None)

pd.set_option('display.max_colwidth', -1)

# import all libraries and dependencies for machine learning

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

from math import isnan

# To perform clustering

from scipy.cluster.hierarchy import cut_tree

from sklearn.metrics import silhouette_samples, silhouette_score

from yellowbrick.cluster import SilhouetteVisualizer

from sklearn.cluster import KMeans

import scipy.cluster.hierarchy as sch

from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.hierarchy import fcluster





class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'

#Data Loading

bank=pd.read_csv('../input/bank-marketing-dataset/bank_marketing_part1_Data.csv')

bank.head()
print(color.BOLD + color.DARKCYAN+'This dataset contains spendings & balance information of customers.')
bank.info()

print("\n")

print("Check for null values:")

print(bank.isnull().sum())

print("\n")



print(color.BOLD + color.DARKCYAN+'Dataset has',bank.shape[0],'rows and',bank.shape[1],'columns.')

print(color.BOLD + color.DARKCYAN+'Dataset has 7 columns of float datatype and there are no null values.')
print(color.BOLD + color.DARKCYAN+'Dataset has',bank.duplicated().sum(),'duplicates.')
bank.describe(include="all")
#Checking for distinct values 

print(color.BOLD+"Unique values per column in dataset :\n")

for column in bank[['spending', 'advance_payments', 'probability_of_full_payment', 'current_balance', 'credit_limit',

                    'min_payment_amt','max_spent_in_single_shopping']]:

    print(color.BOLD+color.DARKCYAN,column.upper(),': ',bank[column].nunique())

    print(color.BOLD+color.DARKCYAN,bank[column].value_counts().sort_values())

    print('\n')
#PLotting Histogram

plt.tight_layout()

bank.hist(figsize=(20,30))

plt.show()
# Let's check the correlation coefficients to see which variables are highly correlated



plt.figure(figsize = (8, 8))

sns.heatmap(bank.corr(), annot = True)

plt.show()
sns.pairplot(bank,diag_kind="kde")

plt.show()
#Checking Outliers with box plot

fig=plt.figure(figsize=(12,9))

for i in range(0,len(bank.columns)):

    ax=fig.add_subplot(3,3,i+1)

    sns.boxplot(y=bank[bank.columns[i]])

    ax.set_title(bank.columns[i],color='Red')

    plt.grid()

plt.tight_layout()

Q1=bank['probability_of_full_payment'].quantile(0.25)

Q3=bank['probability_of_full_payment'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

bank['probability_of_full_payment']=np.where(bank['probability_of_full_payment']>ur,ur,bank['probability_of_full_payment'])

bank['probability_of_full_payment']=np.where(bank['probability_of_full_payment']<lr,lr,bank['probability_of_full_payment'])

   
Q1=bank['min_payment_amt'].quantile(0.25)

Q3=bank['min_payment_amt'].quantile(0.75)

IQR=Q3-Q1

lr = Q1-1.5*IQR

ur = Q3+1.5*IQR

bank['min_payment_amt']=np.where(bank['min_payment_amt']>ur,ur,bank['min_payment_amt'])

bank['min_payment_amt']=np.where(bank['min_payment_amt']<lr,lr,bank['min_payment_amt'])
fig=plt.figure(figsize=(12,9))

for i in range(0,len(bank.columns)):

    ax=fig.add_subplot(3,3,i+1)

    sns.boxplot(y=bank[bank.columns[i]])

    ax.set_title(bank.columns[i],color='Red')

    plt.grid()

plt.tight_layout()

from sklearn.preprocessing import StandardScaler 

X = StandardScaler() 

sc_bank = X.fit_transform(bank)

print(color.BOLD+color.DARKCYAN+'Scaled  dataset:')

pd.DataFrame(sc_bank)
#Plotting dendograms 

x=pd.DataFrame(sc_bank)





Eucildean_Single=sch.linkage(x, method = "single", metric='euclidean')

dend = sch.dendrogram(Eucildean_Single)

plt.title('Dendrogram- Eucildean Single') 

plt.xlabel('Customer Spending') 

plt.ylabel('Euclidean distances')

plt.show()





Manhattan_Single=sch.linkage(x, method = "single", metric='cityblock')

dend = sch.dendrogram(Manhattan_Single)

plt.title('Dendrogram - Manhattan Single') 

plt.xlabel('Users') 

plt.ylabel('Manhattan distances')

plt.show()



Eucildean_Complete=sch.linkage(x, method = "complete", metric='euclidean')

dend = sch.dendrogram(Eucildean_Complete)

plt.title('Dendrogram- Eucildean Complete') 

plt.xlabel('Customer Spending') 

plt.ylabel('Euclidean distances')

plt.show()





Manhattan_Complete=sch.linkage(x, method = "complete", metric='cityblock')

dend = sch.dendrogram(Manhattan_Complete)

plt.title('Dendrogram - Manhattan Complete') 

plt.xlabel('Users') 

plt.ylabel('Manhattan distances')

plt.show()





ward=sch.linkage(x, method = "ward", metric='euclidean')

dend = sch.dendrogram(ward)

plt.title('Dendrogram- Eucildean Ward') 

plt.xlabel('Customer Spending') 

plt.ylabel('Euclidean distances')

plt.show()



dend = dendrogram(ward,truncate_mode='lastp',p = 25)

plt.title('Truncated Cluster Dendogram with Ward Method') 

plt.xlabel('Customer Spending') 

plt.ylabel('Euclidean distances')

plt.show()
clusters = fcluster(ward, 3, criterion='maxclust')

print(color.BOLD+color.DARKCYAN+'Array of clusters from Hierachical Clustering:')

clusters
#Extracting clustered data to csv

bank.to_csv('Hierachical_Wit.csv')
#K-Means clustering is applied on scaled dataset sc_bank for cluster values from 2 to 8 to interpret values for Within sum of sqaures and between sum of squares

wss=[]

import numpy as np 

from scipy.cluster.vq import vq

from tabulate import tabulate

for i in range(2,9):

    x=KMeans(n_clusters=i, random_state=42).fit(sc_bank).cluster_centers_

    partition, euc_distance_to_centroids = vq(sc_bank, x)

    wss=((sc_bank-sc_bank.mean(0))**2).sum(0) 

    TSS = np.sum((sc_bank-sc_bank.mean(0))**2) 

    SSW = np.sum(euc_distance_to_centroids**2) 

    SSB = TSS - SSW 

    WSS=pd.DataFrame([euc_distance_to_centroids**2,KMeans(n_clusters=i, random_state=0).fit(sc_bank).labels_]).T 

    print(tabulate([[i,round(SSB,2),round(TSS,2), np.round(WSS.groupby(1).sum().values,2), round(SSW,2),WSS.groupby(1).count().values]], headers=['Number of Cluster', 'B/w SS','Total SS','Within SS','Total Within SS','Size'])) 

    print('\n')



wcss = [] 

for i in range(2, 9): 

    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42) 

    kmeans.fit(sc_bank) # inertia method returns wcss for that model 

    wcss.append(kmeans.inertia_)

wcss

sns.lineplot(range(2,9), wcss,marker='o',color='red') 

plt.title('Elbow Plot') 

plt.xlabel('Number of clusters') 

plt.ylabel('WCSS') 

plt.show()
#Calculating and plotting silhouette_score

for n_clusters in range(2,9):

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.set_size_inches(20, 10)

    

    

    ax2.set_xlim([-0.1, 1])

    ax2.set_ylim([0, len(x) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=42)

    cluster_labels = clusterer.fit_predict(sc_bank)

    silhouette_avg = silhouette_score(sc_bank, cluster_labels)

    print("For n_clusters =", n_clusters,

          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(sc_bank, cluster_labels)



    for i in range(n_clusters):

        vzr = SilhouetteVisualizer(clusterer)

        vzr.fit(sc_bank)

    ax2.set_title("The silhouette plot for the various clusters.")

    ax2.set_xlabel("The silhouette coefficient values")

    ax2.set_ylabel("Cluster label")

  

    

    

    

    # 2nd Plot showing the actual clusters formed

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

    ax1.scatter(sc_bank[:, 0], sc_bank[:, 1], marker='.', s=30, lw=0, alpha=0.7,

                c=colors, edgecolor='k')



    # Labeling the clusters

    centers = clusterer.cluster_centers_

    # Draw white circles at cluster centers

    ax1.scatter(centers[:, 0], centers[:, 1], marker='o',

                c="white", alpha=1, s=200, edgecolor='k')



    for i, c in enumerate(centers):

        ax1.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,

                    s=50, edgecolor='k')



    ax1.set_title("The visualization of the clustered data.")

    ax1.set_xlabel("Feature space for the 1st feature")

    ax1.set_ylabel("Feature space for the 2nd feature")

    

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "

                  "with n_clusters = %d" % n_clusters),

                 fontsize=14, fontweight='bold')



plt.show()

k_means = KMeans(n_clusters=3, random_state=42)
k_means.fit(sc_bank)

labels=k_means.labels_

print(color.BOLD+color.DARKCYAN+'Array of clusters from Kmeans Clustering:')

labels
#Appending clusters to original dataset

bank['Clusters']=labels

bank.head()
from sklearn.manifold import TSNE



fig, axes = plt.subplots(2,2,figsize=(15,12))

fig.set_size_inches(20, 10)

tsne=TSNE()

visualization=tsne.fit_transform(bank)

sns.set(rc={'figure.figsize':(5,5)})

sns.scatterplot(x=visualization[:,0],y=visualization[:,1],hue=bank['Clusters'],

               palette=sns.color_palette('Set1',3),ax=axes[0][0])





from sklearn.decomposition import PCA

pca_2 = PCA(2) 

plot_columns = pca_2.fit_transform(sc_bank) 

plt.figure(figsize=(12,7)) 

sns.scatterplot(x=plot_columns[:,1], y=plot_columns[:,0], hue=KMeans(n_clusters=3, random_state=42).fit(sc_bank).labels_, palette='Dark2_r',legend=False,ax=axes[0][1]) 

plt.show()
bank.groupby('Clusters').describe()