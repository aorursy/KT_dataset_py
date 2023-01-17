%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import seaborn as sns; sns.set(style='white')



from sklearn import decomposition

from sklearn import datasets

from mpl_toolkits.mplot3d import Axes3D



from pylab import rcParams

rcParams['figure.figsize'] = 12, 12


COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan', 'tab:gray']

MARKERS = ['o', 'v', 's', '<', '>', '8', '^', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']



def plot2d(X, y_pred, y_true, mode=None, centroids=None):

    transformer = None

    X_r = X

    

    if mode is not None:

        transformer = mode(n_components=2)

        X_r = transformer.fit_transform(X)



    assert X_r.shape[1] == 2, 'plot2d only works with 2-dimensional data'





    plt.grid()

    for ix, iyp, iyt in zip(X_r, y_pred, y_true):

        plt.plot(ix[0], ix[1], 

                    c=COLORS[iyp], 

                    marker=MARKERS[iyt])

        

    if centroids is not None:

        C_r = centroids

        if transformer is not None:

            C_r = transformer.fit_transform(centroids)

        for cx in C_r:

            plt.plot(cx[0], cx[1], 

                        marker=MARKERS[-1], 

                        markersize=10,

                        c='red')



    plt.show()



def plot3d(X, y_pred, y_true, mode=None, centroids=None):

    transformer = None

    X_r = X

    if mode is not None:

        transformer = mode(n_components=3)

        X_r = transformer.fit_transform(X)



    assert X_r.shape[1] == 3, 'plot2d only works with 3-dimensional data'



    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')

    ax.elev = 30

    ax.azim = 120



    for ix, iyp, iyt in zip(X_r, y_pred, y_true):

        ax.plot(xs=[ix[0]], ys=[ix[1]], zs=[ix[2]], zdir='z',

                    c=COLORS[iyp], 

                    marker=MARKERS[iyt])

        

    if centroids is not None:

        C_r = centroids

        if transformer is not None:

            C_r = transformer.fit_transform(centroids)

        for cx in C_r:

            ax.plot(xs=[cx[0]], ys=[cx[1]], zs=[cx[2]], zdir='z',

                        marker=MARKERS[-1], 

                        markersize=10,

                        c='red')

    plt.show()
plot2d(X, y, y, TSNE)
plot3d(X, y, y, PCA)
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
from __future__ import absolute_import, division, print_function, unicode_literals

import os

import pathlib

import tensorflow as tf

#tf.enable_eager_execution()

from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

%matplotlib inline
# Compatibility operations

from tensorflow.compat.v1 import ConfigProto

from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()

config.gpu_options.allow_growth = True

session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE
import os

import cv2

import skimage

from skimage import transform

import numpy as np



test_dir =  "../input/kermany2018/oct2017/OCT2017 /train/"

imageSize=24



# ['DME', 'CNV', 'NORMAL', '.DS_Store', 'DRUSEN']

from tqdm import tqdm

def get_data(folder):

    """

    Load the data and labels from the given folder.

    """

    X = []

    y = []

    #max_images = 10

    for folderName in os.listdir(folder):

        #print(folderName)

        max_images_per_category = 700

        if not folderName.startswith('.'):

            if folderName in ['NORMAL']:

                label = 0

            elif folderName in ['CNV']:

                label = 1

            elif folderName in ['DME']:

                label = 2

            elif folderName in ['DRUSEN']:

                label = 3

            else:

                label = 4

            #print("label", label)



            for image_filename in tqdm(os.listdir(folder + folderName)):

                #print(image_filename)

                #max_images -= 1

                max_images_per_category -= 1

                if max_images_per_category <= 0:

                    break

                img_file = cv2.imread(folder + folderName + '/' + image_filename)

                if img_file is not None:

                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))

                    img_arr = np.asarray(img_file)

                    img_arr = np.reshape(img_arr, -1)

                    #print("shape of img afetr reshape", img_arr.shape)

                    X.append(img_arr)

                    y.append(label)

                    #print("printam y", y)

            if max_images_per_category <= 0:

                continue

    X = np.asarray(X)

    y = np.asarray(y)

    return X,y

#X_train, y_train = get_data(train_dir) # Un-comment to use full dataset: Step 1 of 2

X, y = get_data(test_dir)
import seaborn

sns.countplot(y);
import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from sklearn import datasets

from sklearn.cluster import KMeans, MeanShift, DBSCAN, Birch

from sklearn import metrics
output = pd.DataFrame(index=['K-Means','DBSCAN'],

                      columns=['ARI','MI','HCV','FM','SC','CH','DB'])
clust_model = KMeans(n_clusters=4, random_state=7)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['K-Means','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['K-Means','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['K-Means','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['K-Means','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['K-Means','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['K-Means','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['K-Means','DB'] = metrics.davies_bouldin_score(X, labels)
plot2d(X, clust_model.labels_, y, mode=PCA, centroids=clust_model.cluster_centers_)
# Fitting DBSCAN to data

clust_model = DBSCAN(min_samples=2, eps=10)

clust_model.fit(X)

# Evaluating model's performance

labels = clust_model.labels_

output.loc['DBSCAN','ARI'] = metrics.adjusted_rand_score(y, labels)

output.loc['DBSCAN','MI'] = metrics.adjusted_mutual_info_score(y, labels)

output.loc['DBSCAN','HCV'] = metrics.homogeneity_score(y, labels)

output.loc['DBSCAN','FM'] = metrics.fowlkes_mallows_score(y, labels)

output.loc['DBSCAN','SC'] = metrics.silhouette_score(X, labels, metric='euclidean')

output.loc['DBSCAN','CH'] = metrics.calinski_harabaz_score(X, labels)

output.loc['DBSCAN','DB'] = metrics.davies_bouldin_score(X, labels)
plot2d(X, clust_model.labels_, y, mode=PCA)
output
print(X[:10, :])

print(y)