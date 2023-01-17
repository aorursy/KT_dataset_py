import matplotlib.pyplot as plt

import numpy as np

from skimage.io import imshow

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
faces_target = np.load('../input/olivetti_faces_target.npy')
# load dataset.

faces_image = np.load('../input/olivetti_faces.npy')



# show dataset.

fig, axes = plt.subplots(3, 4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(faces_image[i], cmap='bone')
# use PCA to do face dimensionality reduction.

n_components = 400



def face_pca(faces, n_components):

    '''

    利用 sklearn 的 PCA 进行图像降维

    faces: 人脸数据集，faces[i] 为一张 64*64 的图片

    n_components: 选择多少个主成分

    return: pca以后的人脸数据，特征脸

    '''

    pca = PCA(n_components, whiten=True, svd_solver='full')

    flatten_faces = faces.reshape(faces.shape[0], 64 * 64)

    # normalize features

    std_flat_faces = np.empty(flatten_faces.shape)

    #for i in range(64*64):

    std_scale = StandardScaler(copy=True, with_mean=True, with_std=True)

    std_flat_faces = std_scale.fit_transform(flatten_faces)

    faces_pca = pca.fit_transform(std_flat_faces)

    eigen_faces = pca.components_.reshape(n_components, 64, 64)

    print('Total explained variance ratio: %s' % pca.explained_variance_ratio_.sum())

    return faces_pca, eigen_faces, pca





faces_pca, eigen_faces, pca = face_pca(faces_image, n_components)



# Show eigen faces

fig, axes = plt.subplots(3,4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(eigen_faces[i], cmap='bone')
# Implement k-means.

np.random.seed(1)



class KMeans():

    def __init__(self, k=10, max_iter=500):

        self.k = k

        self.max_iter = max_iter

        # n * 1 的数组，保存每个样本的 final cluster id

        self.labels = None

    

    def _dist(self, a, b):

        '''

        a: 一维数组

        b: 一维数组

        return: 欧几里得距离

        '''

        return np.math.sqrt(sum(np.power(a - b, 2)))

    

    def _randCenter(self, data, k):

        '''

        data: n * m 的样本，其中 n 是样本个数，m 是特征个数

        k: cluster 的个数

        return: 随机选择的 k 个质心

        '''

        n, m = data.shape

        

        centers = np.empty((k, m))

#         for i in range(m):

#             minVal = min(data[:, i])

#             maxVal = max(data[:, i])

            # centers[:, i] = (minVal + (maxVal - minVal) * np.random.uniform(0, 1, size=(k,1))).flatten() # 质心第i列的值都处于Xi范围之内

        # randomly choose k data point as center

        for i in range(k):

            face_idx = np.random.choice(np.arange(i*10, (i + 1)*10), size=1)

            centers[i, :] = np.mean(faces_pca[i*10: (i + 1)*10,:], axis=0) # data[face_idx[0],:]

        return centers

    

    def fit(self, data):

        '''

        将输入的data进行聚类，并将聚类结果保存在self.label中

        data: n * m 的样本，其中 n 是样本个数，m 是特征个数

        '''

        n = data.shape[0]

        center_assign = np.zeros(n)

        centers = self._randCenter(data, self.k)

        

        for jjj in range(self.max_iter):

            label_changed = False

            

            for i in range(n):

                point = data[i, :]

                min_dist = np.inf

                min_index = -1

                

                # find the center closest to the data point

                for j in range(self.k):

                    center = centers[j, :]

                    dist = self._dist(point, center)

                    if dist < min_dist:

                        min_dist = dist

                        min_index = j

                        

                if center_assign[i] != min_index:

                    label_changed = True

                    center_assign[i] = min_index

                

            if not label_changed:

                break

            # else update centers

            for kk in range(self.k):

                index = np.nonzero(center_assign == kk)[0]

                centers[kk, :] = np.mean(data[index], axis=0)

        print('Finished! Iteractions = ', jjj)

        self.labels = center_assign



# Clustering

cluster_num = 40

cluster = KMeans(k=cluster_num, max_iter=100)

cluster.fit(faces_pca)



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

    for i, ax in enumerate(fig.axes):

        ax.imshow(this_faces[i], cmap='bone')