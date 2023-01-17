import pandas as pd

from pandas import Series,DataFrame

import numpy as np

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans



from scipy.spatial import distance

from scipy.spatial.distance import cdist, pdist

from sklearn.metrics import euclidean_distances

from sklearn.metrics import silhouette_score



# For Visualization

import matplotlib.pyplot as plt

import matplotlib

from math import sqrt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

%matplotlib inline
#data

df=pd.read_csv('../input/indicators_by_company.csv')

df.head(5)
indicators=['Assets','LiabilitiesAndStockholdersEquity',

'StockholdersEquity',

'CashAndCashEquivalentsAtCarryingValue',

'NetCashProvidedByUsedInOperatingActivities',

'NetIncomeLoss',

'NetCashProvidedByUsedInFinancingActivities',

'CommonStockSharesAuthorized',

'CashAndCashEquivalentsPeriodIncreaseDecrease',

'CommonStockValue',

'CommonStockSharesIssued',

'RetainedEarningsAccumulatedDeficit',

'CommonStockParOrStatedValuePerShare',

'NetCashProvidedByUsedInInvestingActivities',

'PropertyPlantAndEquipmentNet',

'AssetsCurrent',

'LiabilitiesCurrent',

'CommonStockSharesOutstanding',

'Liabilities',

'OperatingIncomeLoss' ]
Values=df.loc[df['indicator_id'].isin(indicators),['company_id','indicator_id','2011']]

Values=pd.melt(Values, id_vars=['company_id', 'indicator_id'], var_name='year', value_name='value')

Values=Values.loc[Values['year']=='2011',['company_id','indicator_id','value']].pivot(index='company_id',columns='indicator_id', values='value').dropna()

Values.head(5)
scaler = StandardScaler().fit(Values[indicators])

Values_Scaled = scaler.transform(Values[indicators])
Values_Reduced_2D = PCA(n_components=2).fit_transform(Values_Scaled)
Values_Reduced_3D = PCA(n_components=3).fit_transform(Values_Scaled)
Values_Reduced_6pc = PCA(n_components=6).fit_transform(Values_Scaled)
def Elbow_Method(data):

    K = range(1,50)

    KM = [KMeans(n_clusters=k).fit(data) for k in K]

    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(data, cent, 'euclidean') for cent in centroids]

    cIdx = [np.argmin(D,axis=1) for D in D_k]

    dist = [np.min(D,axis=1) for D in D_k]

    avgWithinSS = [sum(d)/data.shape[0] for d in dist]

    # Total with-in sum of square

    wcss = [sum(d**2) for d in dist]

    tss = sum(pdist(data)**2)/data.shape[0]

    bss = tss-wcss

    



    # elbow curve

    fig = plt.figure()

    ax = fig.add_subplot(111)

    

    kIdx = 7

    ax.plot(K, avgWithinSS, 'b*-')

    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=10, 

    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')

    

    kIdx = 8

    ax.plot(K, avgWithinSS, 'b*-')

    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=10, 

    markeredgewidth=2, markeredgecolor='b', markerfacecolor='None')

    

    kIdx = 9

    ax.plot(K, avgWithinSS, 'b*-')

    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=10, 

    markeredgewidth=2, markeredgecolor='g', markerfacecolor='None')   

    

    plt.grid(True)

    plt.xlabel('Number of clusters')

    plt.ylabel('Average within-cluster sum of squares')

    plt.title('Elbow for KMeans clustering')



    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(K, bss/tss*100, 'b*-')

    plt.grid(True)

    plt.xlabel('Number of clusters')

    plt.ylabel('Percentage of variance explained')

    plt.title('Elbow for KMeans clustering')
Elbow_Method(Values_Scaled)
Elbow_Method(Values_Reduced_6pc)
Elbow_Method(Values_Reduced_3D)
Elbow_Method(Values_Reduced_2D)
# 2D Visualization



kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10)

kmeans.fit(Values_Reduced_2D)



# Step size of the mesh. Decrease to increase the quality of the VQ.

h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].



# Plot the decision boundary. For that, we will assign a color to each

x_min, x_max = Values_Reduced_2D[:, 0].min() - 1, Values_Reduced_2D[:, 0].max() + 1

y_min, y_max = Values_Reduced_2D[:, 1].min() - 1, Values_Reduced_2D[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# Obtain labels for each point in mesh. Use last trained model.

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])



# Put the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(8, 6))

plt.imshow(Z, interpolation='nearest',

           extent=(xx.min(), xx.max(), yy.min(), yy.max()),

           cmap=plt.cm.Paired,

           aspect='auto', origin='lower')



plt.plot(Values_Reduced_2D[:, 0], Values_Reduced_2D[:, 1], 'k.', markersize=2)

# Plot the centroids as a white X

centroids = kmeans.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1],

            marker='x', s=169, linewidths=3,

            color='w', zorder=10)

plt.title('K-means clustering (PCA-reduced data)\n'

          'Centroids are marked with white cross')

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.xlabel("PC-1")

plt.ylabel("PC-2")
fig = plt.figure(1, figsize=(8, 6))

ax = Axes3D(fig, elev=-150, azim=250)

kmeans = KMeans(init='k-means++', n_clusters=7, n_init=10)

kmeans.fit(Values_Reduced_3D)

ax.scatter(Values_Reduced_3D[:, 0], Values_Reduced_3D[:, 1], Values_Reduced_3D[:, 2], 

           c=kmeans.labels_.astype(np.float)

          )

centroids = kmeans.cluster_centers_

ax.scatter(centroids[:, 0], centroids[:, 1],  centroids[:, 2],

            marker='x', s=169, linewidths=3, 

           c='r')

ax.set_title('K-means clustering (PCA-reduced data)\n'

             'Centroids are marked with red cross')

ax.set_xlabel("PC-1")

ax.w_xaxis.set_ticklabels([])

ax.set_ylabel("PC-2")

ax.w_yaxis.set_ticklabels([])

ax.set_zlabel("PC-3")

ax.w_zaxis.set_ticklabels([])
def bic(clusters, centroids):

    num_points = sum(len(cluster) for cluster in clusters)

    num_dims = clusters[0][0].shape[0]

    log_likelihood = _loglikelihood(num_points, num_dims, clusters, centroids)

    num_params = _free_params(len(clusters), num_dims)

    return log_likelihood - num_params / 2.0 * np.log(num_points)





def _free_params(num_clusters, num_dims):

    return num_clusters * (num_dims + 1)





def _loglikelihood(num_points, num_dims, clusters, centroids):

    ll = 0

    for cluster in clusters:

        fRn = len(cluster)

        t1 = fRn * np.log(fRn)

        t2 = fRn * np.log(num_points)

        variance = _cluster_variance(num_points, clusters, centroids) or np.nextafter(0, 1)

        t3 = ((fRn * num_dims) / 2.0) * np.log((2.0 * np.pi) * variance)

        t4 = (fRn - 1.0) / 2.0

        ll += t1 - t2 - t3 - t4

    return ll



def _cluster_variance(num_points, clusters, centroids):

    s = 0

    denom = float(num_points - len(centroids))

    for cluster, centroid in zip(clusters, centroids):

        distances = euclidean_distances(cluster, centroid)

        s += (distances*distances).sum()

    return s / denom





def compute_bic(kmeans,X):

    """

    Computes the BIC metric for a given clusters



    Parameters:

    -----------------------------------------

    kmeans:  List of clustering object from scikit learn



    X     :  multidimension np array of data points



    Returns:

    -----------------------------------------

    BIC value

    """

    # assign centers and labels

    centers = [kmeans.cluster_centers_]

    labels  = kmeans.labels_

    #number of clusters

    m = kmeans.n_clusters

    # size of the clusters

    n = np.bincount(labels)

    #size of data set

    N, d = X.shape



    #compute variance for all clusters beforehand

    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(m)])



    const_term = 0.5 * m * np.log(N) * (d+1)



    BIC = np.sum([n[i] * np.log(n[i]) -

               n[i] * np.log(N) -

             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -

             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term



    return(BIC)



def BIC_Method(data):

    sns.set_style("ticks")

    sns.set_palette(sns.color_palette("Blues_r"))

    bics = []

    for n_clusters in range(2,50):

        kmeans = KMeans(n_clusters=n_clusters)

        kmeans.fit(data)



        labels = kmeans.labels_

        centroids = kmeans.cluster_centers_



        clusters = {}

        for i,d in enumerate(kmeans.labels_):

            if d not in clusters:

                clusters[d] = []

            clusters[d].append(data[i])



        bics.append(compute_bic(kmeans,data))#-bic(clusters.values(), centroids))



    plt.plot(bics)

    plt.ylabel("BIC score")

    plt.xlabel("Number of clusters")

    plt.title("BIC scoring for K-means cell's behaviour")

    sns.despine()
BIC_Method(Values_Scaled)
BIC_Method(Values_Reduced_6pc)
BIC_Method(Values_Reduced_3D)
BIC_Method(Values_Reduced_2D)
def Silhouette_Method(data):

    s = []

    for n_clusters in range(2,50):

        kmeans = KMeans(n_clusters=n_clusters)

        kmeans.fit(data)



        labels = kmeans.labels_

        centroids = kmeans.cluster_centers_



        s.append(silhouette_score(data, labels, metric='euclidean'))



    plt.plot(s)

    plt.ylabel("Silouette")

    plt.xlabel("Number of clusters")

    plt.title("Silouette for K-means cell's behaviour")

    #sns.despine()
Silhouette_Method(Values_Scaled)
Silhouette_Method(Values_Reduced_6pc)
Silhouette_Method(Values_Reduced_3D)
Silhouette_Method(Values_Reduced_2D)