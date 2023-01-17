import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import sparse
from math import sqrt
import time
def _split_node(node, threshold, branching_factor):
    # 初始化两个类簇
    new_subcluster1 = _CFSubcluster()
    new_subcluster2 = _CFSubcluster()
    
    # 初始化两个CF节点
    new_node1 = _CFNode(threshold, branching_factor, is_leaf=node.is_leaf,n_features=node.n_features)
    new_node2 = _CFNode(threshold, branching_factor, is_leaf=node.is_leaf,n_features=node.n_features)
    new_subcluster1.child_ = new_node1
    new_subcluster2.child_ = new_node2

    # 因为所有的叶节点都是分裂出来的,所以在分裂函数里面可以标记所有的叶节点,方便以后查询
    if node.is_leaf:
        if node.prev_leaf_ is not None:
            node.prev_leaf_.next_leaf_ = new_node1
        new_node1.prev_leaf_ = node.prev_leaf_
        new_node1.next_leaf_ = new_node2
        new_node2.prev_leaf_ = new_node1
        new_node2.next_leaf_ = node.next_leaf_
        if node.next_leaf_ is not None:
            node.next_leaf_.prev_leaf_ = new_node2
    
    # 计算所有数据两两之间的距离
    m = len(node.centroids_)
    dist = np.zeros((m,m))
    for i in range(m):
        for j in range(i+1,m):
            dist[i,j] = np.linalg.norm(node.centroids_[i]-node.centroids_[j])
            dist[j,i] = dist[i,j]
            
    n_clusters = dist.shape[0]

    # 找到距离最远的两个点相对其它点的距离
    farthest_idx = np.unravel_index(dist.argmax(), (n_clusters, n_clusters))
    node1_dist, node2_dist = dist[(farthest_idx,)]
    # 找到离node1更近的点的索引
    node1_closer = node1_dist < node2_dist
    
    # 插入两个类
    for idx, subcluster in enumerate(node.subclusters_):
        if node1_closer[idx]:
            new_node1.append_subcluster(subcluster)
            new_subcluster1.update(subcluster)
        else:
            new_node2.append_subcluster(subcluster)
            new_subcluster2.update(subcluster)
            
    return new_subcluster1, new_subcluster2
class _CFNode(object):
    def __init__(self, threshold, branching_factor, is_leaf, n_features):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.is_leaf = is_leaf
        self.n_features = n_features

        # 叶节点下的类簇
        self.subclusters_ = []
        # 预分配一个矩阵,用来存放所有子节点的线性和
        self.init_centroids_ = np.zeros((branching_factor + 1, n_features))
        # 预分配一个矩阵,用来存放所有子节点的平方和
        self.init_sq_norm_ = np.zeros((branching_factor + 1))
        self.squared_norm_ = []
        self.prev_leaf_ = None
        self.next_leaf_ = None

    def append_subcluster(self, subcluster):
        n_samples = len(self.subclusters_)
        self.subclusters_.append(subcluster)
        self.init_centroids_[n_samples] = subcluster.centroid_
        self.init_sq_norm_[n_samples] = subcluster.sq_norm_

        # 设置不为0的线性和与平方和,因为init_centroids_与init_sq_norm_预分配的缘故,数据大多为0
        self.centroids_ = self.init_centroids_[:n_samples + 1, :]
        self.squared_norm_ = self.init_sq_norm_[:n_samples + 1]

    def update_split_subclusters(self, subcluster, new_subcluster1, new_subcluster2):
        # 原来指向subcluster改为指向new_subcluster1
        ind = self.subclusters_.index(subcluster)
        self.subclusters_[ind] = new_subcluster1
        self.init_centroids_[ind] = new_subcluster1.centroid_
        self.init_sq_norm_[ind] = new_subcluster1.sq_norm_
        # 插入new_subcluster2
        self.append_subcluster(new_subcluster2)

    def insert_cf_subcluster(self, subcluster):
        # 如果该CF节点下面没有子节点,直接插入
        if not self.subclusters_:
            self.append_subcluster(subcluster)
            return False

        # 计算距离,寻找最近的类簇
        dist_matrix = np.dot(self.centroids_, subcluster.centroid_)
        dist_matrix *= -2.
        dist_matrix += self.squared_norm_
        closest_index = np.argmin(dist_matrix)
        closest_subcluster = self.subclusters_[closest_index]

        # 如果子集群有子child，递归
        if closest_subcluster.child_ is not None:
            split_child = closest_subcluster.child_.insert_cf_subcluster(subcluster)

            # 如果不需要分裂,更新父节点的信息
            if not split_child:
                closest_subcluster.update(subcluster)
                self.init_centroids_[closest_index] = self.subclusters_[closest_index].centroid_
                self.init_sq_norm_[closest_index] = self.subclusters_[closest_index].sq_norm_
                return False

            # 事情不太妙. 我们需要把subclusters查分成两部分
            else:
                # 把这个类簇拆分成两部分
                new_subcluster1, new_subcluster2 = _split_node(closest_subcluster.child_, self.threshold, self.branching_factor)
                
                self.update_split_subclusters(closest_subcluster, new_subcluster1, new_subcluster2)
                
                # 如果分裂后父节点叶超过限制,继续分裂
                if len(self.subclusters_) > self.branching_factor:
                    return True
                return False

        # closest_subcluster下面没有child了
        else:
            merged = closest_subcluster.merge_subcluster(subcluster, self.threshold)
            
            # 合并成功,更新叶子节点数据记录
            if merged:
                self.init_centroids_[closest_index] =  closest_subcluster.centroid_
                self.init_sq_norm_[closest_index] = closest_subcluster.sq_norm_
                return False

            # 如果不能合并,创建一个新的类簇
            elif len(self.subclusters_) < self.branching_factor:
                self.append_subcluster(subcluster)
                return False

            # 叶子节点所能承载的类簇数量已经到达限制,创建一个类簇插入进去,然后分裂
            else:
                self.append_subcluster(subcluster)
                return True
class _CFSubcluster(object):
    def __init__(self, linear_sum=None):
        if linear_sum is None:
            self.n_samples_ = 0
            self.squared_sum_ = 0.0
            self.linear_sum_ = 0
        else:
            self.n_samples_ = 1
            # 计算线性和
            self.centroid_ = self.linear_sum_ = linear_sum
            # 计算平方和
            self.squared_sum_ = self.sq_norm_ = np.dot(self.linear_sum_, self.linear_sum_)
        self.child_ = None

    def update(self, subcluster):
        self.n_samples_ += subcluster.n_samples_
        self.linear_sum_ += subcluster.linear_sum_
        self.squared_sum_ += subcluster.squared_sum_
        self.centroid_ = self.linear_sum_ / self.n_samples_
        self.sq_norm_ = np.dot(self.centroid_, self.centroid_)

    def merge_subcluster(self, nominee_cluster, threshold):
        """检测是否可以合并,如果可以就合并
        """
        # 计算合并后的CF特征
        new_ss = self.squared_sum_ + nominee_cluster.squared_sum_
        new_ls = self.linear_sum_ + nominee_cluster.linear_sum_
        new_n = self.n_samples_ + nominee_cluster.n_samples_
        # 型心的CF特征
        new_centroid = (1 / new_n) * new_ls
        new_norm = np.dot(new_centroid, new_centroid)
        # 计算半径的平方
        dot_product = (-2 * new_n) * new_norm
        sq_radius = (new_ss + dot_product) / new_n + new_norm
        
        if sq_radius <= threshold ** 2:
            (self.n_samples_, self.linear_sum_, self.squared_sum_,self.centroid_, self.sq_norm_) = new_n, new_ls, new_ss, new_centroid, new_norm
            return True
        return False
class Birch():
    def __init__(self, threshold=0.5, branching_factor=50):
        self.threshold = threshold
        self.branching_factor = branching_factor
        
    def fit(self, X):
        n_samples, n_features = X.shape

        # 创建根节点
        self.root_ = _CFNode(self.threshold, self.branching_factor, is_leaf=True, n_features=n_features)

        # next_leaf_指针用于连接所有的叶节点,方便以后查询.因为没有节点的next_leaf_是root,所以建立一个假节点.
        self.dummy_leaf_ = _CFNode(self.threshold, self.branching_factor, is_leaf=True, n_features=n_features)
        self.dummy_leaf_.next_leaf_ = self.root_
        self.root_.prev_leaf_ = self.dummy_leaf_

        # 对每一个数据建立一个类簇_CFSubcluster,然后逐一插入到CF树中
        for sample in X:
            subcluster = _CFSubcluster(linear_sum=sample)
            split = self.root_.insert_cf_subcluster(subcluster)

            # 如果根节点需要分裂,直接把根节点删掉,重新建,然后插入
            if split:
                new_subcluster1, new_subcluster2 = _split_node(self.root_, self.threshold, self.branching_factor)
                del self.root_
                self.root_ = _CFNode(self.threshold, self.branching_factor, is_leaf=False, n_features=n_features)
                self.root_.append_subcluster(new_subcluster1)
                self.root_.append_subcluster(new_subcluster2)

        # 获取各叶子节点的线性和,也就是聚类中心
        centroids = np.concatenate([leaf.centroids_ for leaf in self._get_leaves()])
        self.subcluster_centers_ = centroids

        return self

    def _get_leaves(self):
        leaf_ptr = self.dummy_leaf_.next_leaf_
        leaves = []
        while leaf_ptr is not None:
            leaves.append(leaf_ptr)
            leaf_ptr = leaf_ptr.next_leaf_
        return leaves
num = 200
# 标准圆形
mean = [10,10]
cov = [[1,0],
       [0,1]] 
x1,y1 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x1,y1,'x')

# 椭圆，椭圆的轴向与坐标平行
mean = [2,10]
cov = [[0.5,0],
       [0,3]] 
x2,y2 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x2,y2,'x')

# 椭圆，但是椭圆的轴与坐标轴不一定平行
mean = [5,5]
cov = [[1,2.3],
       [2.3,1.4]] 
x3,y3 = np.random.multivariate_normal(mean,cov,num).T
plt.plot(x3,y3,'x')

X = np.concatenate((x1,x2,x3)).reshape(-1,1)
Y = np.concatenate((y1,y2,y3)).reshape(-1,1)
data = np.hstack((X, Y))
# 可以合并的半径
threshold=3
# 每个枝节点包含最多叶节点的个数
branching_factor=50

model = Birch(threshold, branching_factor)
model.fit(data)

# 绘制聚类中心
centers = model.subcluster_centers_
plt.plot(data[:,0],data[:,1],'x')
plt.plot(centers[:,0],centers[:,1],'o',c='r')
# 可以合并的半径
threshold=0.6
# 每个枝节点包含最多叶节点的个数
branching_factor=50

model = Birch(threshold, branching_factor)
model.fit(data)

# 绘制聚类中心
centers = model.subcluster_centers_
plt.plot(data[:,0],data[:,1],'x')
plt.plot(centers[:,0],centers[:,1],'o',c='r')
