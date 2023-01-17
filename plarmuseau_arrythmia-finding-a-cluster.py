import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sn



train = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels/DS1_signals.csv", header=None)

labels = pd.read_csv("../input/mitbih-arrhythmia-database-de-chazal-class-labels//DS1_labels.csv", header=None)

train
labels[0].astype('int').values
def plotting(X1,y_pred,clunm,lendat,lenclu,plot_num):

    plt.subplot(lendat,lenclu, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)



    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',

                                         '#f781bf', '#a65628', '#984ea3',

                                         '#999999', '#e41a1c', '#dede00']),

                                      int(max(y_pred) + 1))))

    # add black color for outliers (if any)

    colors = np.append(colors, ["#000000"])

    #print(X1)

    plt.scatter(X1[:, 0], X1[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)

    plt.ylim(-2.5, 2.5)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, clunm,transform=plt.gca().transAxes, size=15,horizontalalignment='right')

    

    return plot_num+1



  

import time

import warnings



import numpy as np

import matplotlib.pyplot as plt



from sklearn import cluster, datasets, mixture

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice



from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

from sklearn.manifold import TSNE,LocallyLinearEmbedding,Isomap,SpectralEmbedding,MDS #limit number of records to 100000

from sklearn.random_projection import SparseRandomProjection

from sklearn.ensemble import RandomTreesEmbedding

n_neighbors=5

nco=3



clusters = [PCA(n_components=nco,random_state=0,whiten=True),

            SparseRandomProjection(random_state=27,n_components=nco),

             NeighborhoodComponentsAnalysis(n_components=nco,random_state=0),

             TruncatedSVD(n_components=nco, n_iter=7, random_state=42),

             #NMF(n_components=5,random_state=0),            

             #UMAP(n_neighbors=5,n_components=nco, min_dist=0.3,metric='minkowski'),

             TSNE(n_components=nco,random_state=0),

             MDS(n_components=nco, n_init=1, max_iter=100),

             FastICA(n_components=nco,random_state=0),

             #RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5) 

            #LocallyLinearEmbedding(n_neighbors, n_components=nco,  method='standard'),

            #LinearDiscriminantAnalysis(n_components=nco),

            #Isomap(n_neighbors, n_components=nco),

            #SpectralEmbedding(n_components=nco, random_state=0,eigen_solver="arpack"),



           ] 

clunaam=['PCA','Spar','nCA','tSVD','tSNE','MDS','fICA','rTremb'] #,'fICA','rTrEmb',LocLinEmb','LDA','Isomap' ,'Spect',

 

np.random.seed(0)



# ============

# Clustering with PCA, SVD etc

# ============

plt.figure(figsize=(11* 2 + 3, 12.5))

plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,

                    hspace=.01)



plot_num = 1



default_base = {'quantile': .3,

                'eps': .3,

                'damping': .9,

                'preference': -200,

                'n_neighbors': 10,

                'n_clusters': 3,

                'min_samples': 20,

                'xi': 0.05,

                'min_cluster_size': 0.1}



datasets = clunaam



label='Survived'

indexv='PassengerId'

dtrain=train



lenxtr=len(dtrain)

i_dataset=0

for clu in clusters:

    clunm=clunaam[clusters.index(clu)] #find naam

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(dtrain.iloc[:2000])    

    y = labels[0].iloc[:2000].values    

    X = clu.fit_transform(X,y)



    print(X.shape)

    #plt.scatter(X[:lenxtr,0],X[:lenxtr,1],c=y.values,cmap='prism')

    #plt.title(clu)

    #plt.show()

    #for i_dataset, (dataset, algo_params) in enumerate(datasets):

    # update parameters with dataset-specific values

    params = default_base.copy()

    params.update([])



    # estimate bandwidth for mean shift

    bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])



    # connectivity matrix for structured Ward

    connectivity = kneighbors_graph(

        X, n_neighbors=params['n_neighbors'], include_self=False)

    # make connectivity symmetric

    connectivity = 0.5 * (connectivity + connectivity.T)



    # ============

    # Create cluster objects

    # ============

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])

    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)

    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack',affinity="nearest_neighbors")

    dbscan = cluster.DBSCAN(eps=params['eps'])

    optics = cluster.OPTICS(min_samples=params['min_samples'],xi=params['xi'],min_cluster_size=params['min_cluster_size'])

    affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])

    average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=params['n_clusters'], connectivity=connectivity)

    birch = cluster.Birch(n_clusters=params['n_clusters'])

    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')



    clustering_algorithms = (

        ('MiniBatchKMeans', two_means),

        ('AffinityPropagation', affinity_propagation),

        ('MeanShift', ms),

        ('SpectralClustering', spectral),

        ('Ward', ward),

        ('AgglomerativeClustering', average_linkage),

        ('DBSCAN', dbscan),

        ('OPTICS', optics),

        ('Birch', birch),

        ('GaussianMixture', gmm)

    )

    name='survived'

    plotting(X,y,clunm,len(datasets), len(clustering_algorithms)+1,plot_num)

    plot_num+=1

    for name, algorithm in clustering_algorithms:

        t0 = time.time()



        # catch warnings related to kneighbors_graph

        with warnings.catch_warnings():

            warnings.filterwarnings(

                "ignore",

                message="the number of connected components of the " +

                "connectivity matrix is [0-9]{1,2}" +

                " > 1. Completing it to avoid stopping the tree early.",

                category=UserWarning)

            warnings.filterwarnings(

                "ignore",

                message="Graph is not fully connected, spectral embedding" +

                " may not work as expected.",

                category=UserWarning)

            algorithm.fit(X)



        t1 = time.time()

        if hasattr(algorithm, 'labels_'):

            y_pred = algorithm.labels_.astype(np.int)

        else:

            y_pred = algorithm.predict(X)

        plot_num=plotting(X,y_pred,clunm,len(datasets), len(clustering_algorithms)+1,plot_num)

    i_dataset+=1



plt.show()