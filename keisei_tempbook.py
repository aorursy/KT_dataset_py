import time



import numpy as np

import matplotlib.pyplot as plt



from sklearn import cluster

from sklearn.neighbors import kneighbors_graph

from sklearn.preprocessing import StandardScaler

from pylab import rcParams

#rcParams['figure.figsize'] = 20, 10

np.random.seed(0)



# Generate datasets. We choose the size big enough to see the scalability

# of the algorithms, but not too big to avoid too long running times

n_samples = 500



colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])

colors = np.hstack([colors] * 20)



plt.figure(figsize=((len(clustering_names) * 2 + 3)*2, 9.5*2))

plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,

                    hspace=.01)
datasets = [[[0,0]],

            [[0,0]],

            [[0,0]],

            [[0,0]]]

mu=0.3



centers=[[(0,0.45),(0.9,0.5),(0.45,0.9),(0.45,0)],

        [(0,0.2),(0.8,0),(0.2,1),(1,0.8)],

        [(0,0),(0.9,0.9),(0,0.9),(0.9,0)],

        [(0,0),(0.9,0),(0.45,0.779),(0.45,0.259)]]



for i,c in enumerate(centers):

    for x,y in c:

        num1 = np.random.normal(x, mu, n_samples)

        num2 = np.random.normal(y, mu, n_samples)

        nums = np.vstack((num1,num2)).T

        datasets[i]=np.vstack((datasets[i],nums))

datasets = list(zip(datasets,['a','b','c','d']))



plot_num = 1

for (X,lbl), c in zip(datasets,centers):

    

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], s=10)

    center_colors = colors[:len(centers)]

    plt.scatter(list(zip(*c))[0], list(zip(*c))[1], s=100, c=center_colors)

    plt.xlim(-1, 2)

    plt.ylim(-1, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, lbl = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)



    # create clustering estimators

    two_means = cluster.MiniBatchKMeans(n_clusters=4)

   

    algorithm=two_means

    name='MiniBatchKMeans'

    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)



    # estimate bandwidth for mean shift

    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)



    algorithm=ms

    name='MeanShift'

    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)



    # connectivity matrix for structured Ward

    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # make connectivity symmetric

    connectivity = 0.5 * (connectivity + connectivity.T)



    ward = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward',

                                           connectivity=connectivity)



    algorithm=ward

    name='Ward'

    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)



    spectral = cluster.SpectralClustering(n_clusters=4,

                                          eigen_solver='arpack',

                                          affinity="nearest_neighbors")

        

    algorithm=spectral

    name='SpectralClustering'



    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)



    dbscan = cluster.DBSCAN(eps=.23,min_samples=30)

    

    algorithm=dbscan

    name='DBSCAN'

    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
plot_num = 1

for i_dataset, dataset in enumerate(datasets):

    X, y = dataset

    # normalize dataset for easier parameter selection

    X = StandardScaler().fit_transform(X)

    '''

    # estimate bandwidth for mean shift

    bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)



    # connectivity matrix for structured Ward

    connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

    # make connectivity symmetric

    connectivity = 0.5 * (connectivity + connectivity.T)



    # create clustering estimators

    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

    two_means = cluster.MiniBatchKMeans(n_clusters=4)

    ward = cluster.AgglomerativeClustering(n_clusters=4, linkage='ward',

                                           connectivity=connectivity)

    spectral = cluster.SpectralClustering(n_clusters=4,

                                          eigen_solver='arpack',

                                          affinity="nearest_neighbors")

    dbscan = cluster.DBSCAN(eps=.23,min_samples=30)

    affinity_propagation = cluster.AffinityPropagation(damping=.9,

                                                       preference=-200)



    average_linkage = cluster.AgglomerativeClustering(

        linkage="average", affinity="cityblock", n_clusters=4,

        connectivity=connectivity)



    birch = cluster.Birch(n_clusters=4)

    clustering_names = [

        'MiniBatchKMeans', 'AffinityPropagation', 'MeanShift',

        'SpectralClustering', 'Ward', 'AgglomerativeClustering',

        'DBSCAN', 'Birch']

    clustering_algorithms = [

        two_means, affinity_propagation, ms, spectral, ward, average_linkage,

        dbscan, birch]

        

        

    algorithm=

    name=''

    '''

    

    # predict cluster memberships

    t0 = time.time()

    algorithm.fit(X)

    t1 = time.time()

    if hasattr(algorithm, 'labels_'):

        y_pred = algorithm.labels_.astype(np.int)

    else:

        y_pred = algorithm.predict(X)



    # plot

    plt.subplot(2, 2, plot_num)

    if i_dataset == 0:

        plt.title(name, size=18)

    plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)



    if hasattr(algorithm, 'cluster_centers_'):

        centers = algorithm.cluster_centers_

        center_colors = colors[:len(centers)]

        plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)

    plt.xlim(-2, 2)

    plt.ylim(-2, 2)

    plt.xticks(())

    plt.yticks(())

    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='right')

    plt.text(.01, .01, lbl,

             transform=plt.gca().transAxes, size=15,

             horizontalalignment='left')

    plot_num += 1



plt.show()
datasets = [([[0,0]],'a'),

            ([[0,0]],'b'),

            ([[0,0]],'c'),

            ([[0,0]],'d'),]

mu=0.3



centers=[[(0,0.45),(0.9,0.5),(0.45,0.9),(0.45,0)],

        [(0,0.2),(0.8,0),(0.2,1),(1,0.8)],

        [(0,0),(0.9,0.9),(0,0.9),(0.9,0)],

        [(0,0),(0.9,0),(0.45,0.779),(0.45,0.259)]]



for (d,_),l in zip(datasets,centers):

    print (d,lb,l)