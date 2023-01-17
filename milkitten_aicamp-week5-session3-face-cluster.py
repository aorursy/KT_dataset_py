import matplotlib.pyplot as plt

import numpy as np

from skimage.io import imshow

from sklearn.decomposition import PCA

face_image = np.load('../input/olivetti_faces.npy')

target = np.load('../input/olivetti_faces_target.npy')



print('image dimension ', face_image.shape)

print('target dimension', target.shape)

print('total unique people', np.unique(target).shape)



# Idx = np.random.choice(range(face_image.shape[0]), 12)

# for i, idx in enumerate(Idx):

#     plt.subplot(3,4,i+1)

#     plt.imshow(face_image[idx])

# plt.show()    



# load dataset.

faces_image = np.load('../input/olivetti_faces.npy')



# show dataset.

fig, axes = plt.subplots(3, 4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(faces_image[i], cmap='bone')
# use PCA to do face dimensionality reduction.

n_components = 150

def face_pca(faces, n_components):

    '''

    利用 sklearn 的 PCA 进行图像降维

    faces: 人脸数据集，faces[i] 为一张 64*64 的图片

    n_components: 选择多少个主成分

    return: pca以后的人脸数据，特征脸

    '''

    h = faces.shape[1]

    w = faces.shape[2]

    faces = faces.reshape(faces.shape[0], -1)

    pca = PCA(n_components = n_components).fit(faces)

    eigen_faces = pca.components_.reshape(n_components,h,w)

    faces_pca = pca.transform(faces)

    return faces_pca, eigen_faces

faces_pca, eigen_faces = face_pca(faces_image, n_components)



print('Original dimension', faces_image.shape)

print('After PCA transform', faces_pca.shape)

print('PCA/eigen_faces', eigen_faces.shape)

# Show eigen faces

fig, axes = plt.subplots(3, 4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(eigen_faces[i], cmap='bone')
# Implement k-means.

# Implement k-means.

class KMeans():

    def __init__(self, k, max_iter):

        self.k = k

        self.max_iter = max_iter

        # n * 1 的数组，保存每个样本的 final cluster id

        self.label = None

    

    def _dist(self, a, b):

        '''

        a: 一维数组

        b: 一维数组

        return: 欧几里得距离

        '''

        diff = a-b

        return np.math.sqrt(np.dot(diff,diff))

    

    def _randCenter(self, data, k):

        '''

        data: n * m 的样本，其中 n 是样本个数，m 是特征个数

        k: cluster 的个数

        return: 随机选择的 k 个质心

        '''

        m = data.shape[1]

        centers = np.empty((k, m))

        for i in range(m):

            minVal = min(data[:, i])

            maxVal = max(data[:, i])

            centers[:, i] = (minVal + (maxVal - minVal) * np.random.rand(k, 1)).flatten()

        return centers

    

    def fit(self, data):

        '''

        将输入的data进行聚类，并将聚类结果保存在self.label中

        data: n * m 的样本，其中 n 是样本个数，m 是特征个数

        '''

        centers = self._randCenter(data,self.k)

        dist_min = np.inf

        cluster_assign = np.full(data.shape[0], -1)

        cluster_dmin = np.full(data.shape[0], np.inf)



        for it in range(self.max_iter):

            cluster_change = False

            for n in range(data.shape[0]):

                assign_idx = -1

                dist_min = np.inf

                for i in range(centers.shape[0]):

                    dist = self._dist(centers[i, :], data[n, :])

                    if dist < dist_min:

                        dist_min = dist

                        assign_idx = i

                if(cluster_assign[n]!=assign_idx and cluster_dmin[n] > dist_min):

                    cluster_assign[n] = assign_idx

                    cluster_dmin[n] = dist_min

                    cluster_change = True    

            if(cluster_change == False):   

                break;

            for i in range(centers.shape[0]):

                index = np.nonzero(cluster_assign == i)[0]

                centers[i,:] = np.mean(data[index],axis=0)

        self.labels = cluster_assign    

# # Clustering

cluster_num = 40

max_iter = 500

cluster = KMeans(cluster_num, max_iter)

cluster.fit(faces_pca)



# Show the final results

# Show the final results

labels = cluster.labels

for i in range(cluster_num):

    index = np.nonzero(labels==i)[0]

    num = len(index)

    this_faces = faces_image[index]

    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4),

                             subplot_kw={'xticks':[], 'yticks':[]},

                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    fig.suptitle("Cluster " + str(i), fontsize=20)

    for i, ax in enumerate(axes.flat):

        ax.imshow(this_faces[i], cmap='bone')
from sklearn.metrics import accuracy_score

correct = np.array([target == labels])

correct.mean()