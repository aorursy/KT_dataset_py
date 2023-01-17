import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from sklearn.decomposition import PCA
import sklearn.cluster as skcluster
# load dataset.
faces_image = np.load('../input/olivetti_faces.npy')

# show dataset.
fig, axes = plt.subplots(3, 4, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_image[i], cmap='bone')
faces_image.shape
# use PCA to do face dimensionality reduction.
n_components = 400
def face_pca(faces, n_components):
    '''
    利用 sklearn 的 PCA 进行图像降维
    faces: 人脸数据集，faces[i] 为一张 64*64 的图片
    n_components: 选择多少个主成分
    return: pca以后的人脸数据，特征脸
    '''
    faces_1d = faces.reshape(faces.shape[0], -1)
    pca = PCA(n_components=n_components).fit(faces_1d)
    eigen_faces = pca.components_.reshape(n_components, faces.shape[1], faces.shape[2])
    faces_pca = pca.transform(faces_1d)
    return faces_pca, eigen_faces

faces_pca, eigen_faces = face_pca(faces_image, n_components)
print(faces_pca.shape)

# Show eigen faces
fig, axes = plt.subplots(3, 4, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigen_faces[i], cmap='bone')
# Implement k-means.
# Implement k-means.
class KMeans():
    def __init__(self, k=40, max_iter=500):
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
        return np.math.sqrt(sum(np.power(a - b, 2)))
    
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
            # centers[:, i] = (minVal + (maxVal - minVal) * np.random.rand(k)
            centers[:, i] = (minVal + (maxVal - minVal) * np.random.rand(k, 1)).flatten()
        return centers
    
    def _find_single_id(self, single_data, centers):
        '''
        Find label for single data
        return label and distance to the label center
        '''
        min_dis = np.inf
        label = -1
        for i, center in enumerate(centers):
            dis = self._dist(center, single_data)
            if dis < min_dis:
                min_dis = dis
                label = i
        return label, min_dis
                
    
    def _find_all_id(self, data_id, data_dis, data, centers):
        '''
        Find labels for all data
        Update data distances to the label center
        return if the centers should be changed
        '''
        n = len(data_id)
        cluster_changed = False
        for i in range(n):
            single_data = data[i]
            single_id, dis = self._find_single_id(single_data, centers)
            
            if single_id != data_id[i] or abs(data_dis[i] - dis) > 10 ** (-5):
                cluster_changed = True
            data_id[i] = single_id
            data_dis[i] = dis
        return cluster_changed
    
    def _update_centers(self, data_id, data, centers):
        for i in range(len(centers)):
            index = np.nonzero(data_id == i)
            centers[i] = np.mean(data[np.nonzero(data_id == i)], axis = 0)
        return

    def fit(self, data):
        '''
        将输入的data进行聚类，并将聚类结果保存在self.label中
        data: n * m 的样本，其中 n 是样本个数，m 是特征个数
        '''
        num_data = data.shape[0]
        data_id = np.zeros(num_data)
        data_dis = np.full(num_data, np.inf)
        
        centers = self._randCenter(data, self.k)
        
        for _ in range(self.max_iter):
            cluster_changed = self._find_all_id(data_id, data_dis, data, centers)
            if not cluster_changed:
                break
            self._update_centers(data_id, data, centers)
        self.labels = data_id
        return
# Clustering
cluster_num = 40
cluster = KMeans(k=cluster_num, max_iter=1000)
cluster.fit(faces_pca)
def show_faces(faces_image, labels, cluster_num):
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
        if num == 1:
            axes.imshow(this_faces[0], cmap='bone')
            continue
        for i_img, ax in enumerate(axes.flat):
            ax.imshow(this_faces[i_img], cmap='bone')
    plt.show()
    return
# Show the final results
labels = cluster.labels
show_faces(faces_image, labels, cluster_num)
cluster_num = 40
cluster = skcluster.KMeans(n_clusters=cluster_num, max_iter=1000)
cluster.fit(faces_pca)
labels = cluster.labels_
show_faces(faces_image, labels, cluster_num)
# The Elbow Method
# Find Best cluster_num

def find_mean_dis(data, labels, centers):
    norm_arrray = np.linalg.norm(data - centers[labels], axis = 1)
    return norm_arrray.sum() / len(norm_arrray)

mean_dis = []
cluster_num_list = range(1, 60)
for cluster_num in cluster_num_list:
    print ('cluster_num:', cluster_num)
    cluster = skcluster.KMeans(n_clusters=cluster_num, max_iter=1000)
    cluster.fit(faces_pca)
    labels = cluster.labels_
    centers = cluster.cluster_centers_
    mean_dis.append(find_mean_dis(faces_pca, labels, centers))

plt.scatter(cluster_num_list, mean_dis)

# It is not that obvious to find best cluster_num whith The Elbow Method???