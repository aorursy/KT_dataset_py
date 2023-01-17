import matplotlib.pyplot as plt

import numpy as np

from skimage.io import imshow

from sklearn.decomposition import PCA



import warnings

warnings.filterwarnings("ignore")

import time
# load dataset.

faces_image = np.load('../input/olivetti_faces.npy')



# show dataset.

fig, axes = plt.subplots(3, 4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(faces_image[i], cmap='bone')

print(faces_image.shape)
# use PCA to do face dimensionality reduction.

n_components = 250

def face_pca(faces, n_components):

    '''

    利用 sklearn 的 PCA 进行图像降维

    faces: 人脸数据集，faces[i] 为一张 64*64 的图片

    n_components: 选择多少个主成分

    return: pca以后的人脸数据，特征脸

    '''

    faces_ravel_data = faces.reshape(faces.shape[0], faces.shape[1] * faces.shape[2])

    pca = PCA(n_components = n_components).fit(faces_ravel_data)

    

    return pca.transform(faces_ravel_data), pca.components_.reshape((n_components, faces.shape[1], faces.shape[2]))

    



faces_pca, eigen_faces = face_pca(faces_image, n_components)



# Show eigen faces

fig, axes = plt.subplots(3, 4, figsize=(9, 4),

                         subplot_kw={'xticks':[], 'yticks':[]},

                         gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(eigen_faces[i], cmap='bone')
class myKMeans():

    def __init__(self, k = 40, max_iter = 500):

        self.k = k

        self.max_iter = max_iter

        self.labels = None

        

    def _random_centers(self, data):

        k = self.k

        m = data.shape[1]

        centers = np.zeros((k, m))

        

        min_data = np.array([min(data[:, i]) for i in range(data.shape[1])])

        max_data = np.array([max(data[:, i]) for i in range(data.shape[1])])

        for i in range(k):

            random_base = np.random.rand(centers.shape[1])

            centers[i] += min_data +  random_base * (max_data - min_data)

        return centers

    

    def _dist(self, a, b):

        #return np.math.sqrt(sum((a-b)**2))

        return np.math.sqrt(sum(np.power(a-b,2)))

    

    def fit(self, data):

        centers = self._random_centers(data)

        cluster_indexes = np.full(data.shape[0], -1)

        cluster_min_dist = np.full(data.shape[0], np.inf)

        

        for iteration in range(self.max_iter):

            cluster_changed = False

            for i in range(data.shape[0]):

                min_dist = np.inf

                assign_cluster_index = -1

                for k in range(self.k):

                    dist = self._dist(centers[k], data[i])

                    if dist < min_dist:

                        assign_cluster_index = k

                        min_dist = dist

                        

                if cluster_indexes[i] != assign_cluster_index and min_dist < cluster_min_dist[i]:

                    cluster_indexes[i] = assign_cluster_index

                    cluster_changed = True

                    cluster_min_dist[i] = min_dist

                    

            if not cluster_changed: 

                break

            

            for k in range(self.k):

                index = np.nonzero(cluster_indexes==k)[0]

                centers[k, :] = np.mean(data[index], axis=0)

                

        self.labels = cluster_indexes

    

    
training_times = []

components = list(range(20, faces_image.shape[0] + 1, 20))

for n_components in components:

    faces_pca, eigen_faces = face_pca(faces_image, n_components)



    cluster_num = 40

    cluster = myKMeans(k = cluster_num)

    start = time.time()

    cluster.fit(faces_pca)

    end = time.time()

    training_time = end - start

    print("Eigen Faces num=" + str(n_components) +", Used Training Time=" + str(training_time) + " secs")

    training_times.append(training_time)

    
# Observe PCA effect and pick the most likely best efficient N of component number

components = np.array(components)

training_times = np.array(training_times)



plt.plot(components, training_times, '-ro')

plt.title("Used Time by N components")

plt.xlabel("components")

plt.ylabel("secs")

plt.show()
n_components = 200

faces_pca, eigen_faces = face_pca(faces_image, n_components)



cluster_num = 40

cluster = myKMeans(k = cluster_num)

start = time.time()

cluster.fit(faces_pca)

end = time.time()

training_time = end - start

print("Eigen Faces num=" + str(n_components) +", Used Training Time=" + str(training_time) + " secs")      



labels = cluster.labels

for i in range(cluster_num):

    index = np.nonzero(labels==i)[0]

    num = len(index)

    if num == 0:

        continue

    this_faces = faces_image[index]

    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4),

                             subplot_kw={'xticks':[], 'yticks':[]},

                             gridspec_kw=dict(hspace=0.1, wspace=0.1))

    fig.suptitle("Cluster " + str(i), fontsize=20)

    

    for i, ax in enumerate(axes.flat):

        ax.imshow(this_faces[i], cmap='bone')