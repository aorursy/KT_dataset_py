import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
# Load dataset.
faces_image = np.load('../input/olivetti_faces.npy')
faces_image[:1].shape
faces_image.shape

# Show dataset.
fig, axes = plt.subplots(3, 4, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_image[i], cmap='bone')
# Use PCA to do face dimensionality reduction.
n_components = 90
def face_pca(faces, n_components):
    '''
    利用 sklearn 的 PCA 进行图像降维
    faces: 人脸数据集，faces[i] 为一张 64*64 的图片
    n_components: 选择多少个主成分
    return: pca以后的人脸数据，特征脸
    '''
    h = faces.shape[1] # i.e.64  each face image: 1 * 64 * 64 (faces.shape[p0] 400. i.e. 400 face images)
    w = faces.shape[2] # i.e. 64
    print(faces.shape)
    # reshape 把第二维和第三维合并
    faces_data = faces.reshape(faces.shape[0], faces.shape[1] * faces.shape[2]) # 400, 64 * 64
    print(faces_data.shape)
    pca = PCA(n_components=n_components).fit(faces_data) # PCA is lib from sklearn package
    
    eigen_faces = pca.components_.reshape((n_components, h, w)) # i.e. 90, 64, 64 也就是特征脸 90 就是对应之前设置的n_components=90,也就是特征脸
    print(eigen_faces.shape)
    faces_pca = pca.transform(faces_data) # 降维以后的数据
    print(faces_pca.shape)
    return faces_pca, eigen_faces

faces_pca, eigen_faces = face_pca(faces_image, n_components)

# Show eigen faces
fig, axes = plt.subplots(3, 4, figsize=(9, 4),
                         subplot_kw={'xticks':[], 'yticks':[]},
                         gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigen_faces[i], cmap='bone')
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
    
    #???
    def _randCenter(self, data, k):
        '''
        data: n * m 的样本，其中 n 是样本个数，m 是特征个数
        k: cluster 的个数
        return: 随机选择的 k 个质心
        '''
        m = data.shape[1]
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.empty.html
        
        centers = np.empty((k, m))
        # 计算逻辑是：K个中心， 每次计算K个中心的其中一个feature
        for i in range(m): # all features
            minVal = min(data[:, i]) # each feature's min value
            maxVal = max(data[:, i]) # each feature's max value
            # np.random.rand(k, 1)  -> generate k row, 1 column value
            centers[:, i] = (minVal + (maxVal - minVal) * np.random.rand(k, 1)).flatten()
        return centers
    
    def fit(self, data):
        '''
        将输入的data进行聚类，并将聚类结果保存在self.label中
        data: n * m 的样本，其中 n 是样本个数，m 是特征个数
        '''
        n = data.shape[0] # 样本个数
        cluster_assign = np.zeros(n) # 样本分配的 cluster id
        # numpy.inf https://www.numpy.org/devdocs/reference/constants.html
        cluster_dis = np.full(n, np.inf) # 样本到 cluster 中心的距离. n * 1, 每个item的值都是 numpy infinity
        
        centers = self._randCenter(data, self.k)
        
        # 控制最大循环次数 500
        for _ in range(self.max_iter):
            cluster_changed = False
            # 遍历所有输入数据
            for i in range(n):
                min_dist = np.inf
                min_index = -1
                # 遍历所有的 cluster 中心，尝试为当前样本分配更近的 cluster
                for j in range(self.k):
                    center = centers[j, :]
                    sample = data[i, :]
                    dis = self._dist(center, sample)
                    if dis < min_dist:
                        min_dist = dis
                        min_index = j
                
                if cluster_assign[i] != min_index and cluster_dis[i] > min_dist:
                    cluster_changed = True
                    cluster_assign[i] = min_index 
                    cluster_dis[i] = min_dist
            # 如果所有样本都没有变化，说明收敛，则退出
            if not cluster_changed:
                break
            
            for i in range(self.k):
                index = np.nonzero(cluster_assign==i)[0]
                print("index======")
                print(index)
                print("mean")
                print(np.mean(data[index], axis=0))
                centers[i, :] = np.mean(data[index], axis=0) # data[index] 原始数据的若干行， axis=0 表示在列上面求均值
        
        self.labels = cluster_assign

# Clustering
cluster_num = 40
cluster = KMeans(k=cluster_num)
cluster.fit(faces_pca)

# Show the final results
labels = cluster.labels # labels.shape: 400, i.e. class for each face image
print(labels.shape)
print(labels) # labels: array with 400 items. each one represent result for a face image
print("==================")
for i in range(cluster_num):
    print(np.nonzero(labels==i))
    index = np.nonzero(labels==i)[0] # from 400 images, find out image indice belong to each class. starting from class 0
    # reason we need [0]. because returned index format: array([ 80,  82, 111, 112, 113, 114, 116]),)
    num = len(index)
    this_faces = faces_image[index] # face images for a class (array)
    # 1 mean 1 row, num: columns
    fig, axes = plt.subplots(1, num, figsize=(4 * num, 4),
                             subplot_kw={'xticks':[], 'yticks':[]},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    fig.suptitle("Cluster " + str(i), fontsize=20)
    # axes.flat???
    for i, ax in enumerate(axes.flat):
        ax.imshow(this_faces[i], cmap='bone')