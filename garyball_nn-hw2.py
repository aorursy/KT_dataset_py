import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#  tqdm 为显示训练进度的库
from tqdm import tqdm
from tqdm._tqdm import trange

import numpy as np
import matplotlib.pyplot as plt
import math
import time
data = pd.read_excel('../input/nn-hw1-data/data_hw1.xlsx', sheet_name=0)
data.head()
data_p2 = data.iloc[:,[0,2]]
data_p2.head()
data_p2.iloc[:,0] = data_p2.iloc[:,0].apply(lambda x:x[0:10]) 
data_p2.head()
# 为数据添加一列序号
array = np.arange(0,288,1)
array1 = np.arange(0,288,1)
for i in range(int(len(data_p2)/288)-1):
    array1 = np.append(array1,array)
data_p2['index'] = array1
data_p2.head()
date = data_p2['Datetime \ Milepost'].unique()
array = np.zeros((31, 288))
i=0
for item in date:
    array[i,] = np.array(data_p2[data_p2['Datetime \ Milepost'] == item].iloc[:,1])
    i = i+1
array.shape
# 标准化
def normalize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    return (x - mean) / std, mean,std

array1, mean, std= normalize(array)
# 定义欧式距离，用于计算两点之间的距离，在后续中和邻域阈值eps进行比较 
UNCLASSIFIED = False
NOISE = 0

def dist(a, b):
    """
    输入：向量a,b
    输出：两个向量的欧式距离
    """
    # print(math.sqrt(np.power(a - b, 2).sum()))
    return math.sqrt(np.power(a - b, 2).sum())
#  判断输入的两个点是否密度可达
def eps_neighbor(a, b, eps):
    """
    输入：向量a,b,eps
    输出：是否在eps范围内
    """
    return dist(a, b) < eps
# 记录ID为 pointID的点eps范围内其他密度可达点的id
def region_query(data, pointId, eps):
    """
    输入：数据集, 查询点id, 半径大小
    输出：在eps范围内的点的id
    """
    nPoints = data.shape[1]
    seeds = []
    for i in range(nPoints):
        if eps_neighbor(data[:, pointId], data[:, i], eps):
            seeds.append(i)
    return seeds
# 首先判断pointID是否是核心点，如果是核心点，首先将新的核心点划分进前一点的cluster之内， 
def expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
    """
    输入：数据集, 分类结果, 待分类点id, 簇id, 半径大小, 最小点个数
    输出：能否成功分类
    """
    seeds = region_query(data, pointId, eps)
    # 不满足minPts条件设置为噪声点，分类标签NOISE = 0
    if len(seeds) < minPts: 
        clusterResult[pointId] = NOISE
        return False
    #  如果 pointID是核心点，进行后续操作
    else:
        clusterResult[pointId] = clusterId # 划分到该簇
        for seedId in seeds:
            clusterResult[seedId] = clusterId
 
        # 在邻域内的其他点，
        while len(seeds) > 0: # 持续判断
            currentPoint = seeds[0]
            queryResults = region_query(data, currentPoint, eps)
            # 如果当前的点是核心
            if len(queryResults) >= minPts:
                for i in range(len(queryResults)):
                    resultPoint = queryResults[i]
                    if clusterResult[resultPoint] == UNCLASSIFIED:
                        seeds.append(resultPoint)
                        clusterResult[resultPoint] = clusterId
                    elif clusterResult[resultPoint] == NOISE:
                        clusterResult[resultPoint] = clusterId
            # 去除当前已经遍历过的点
            seeds = seeds[1:]
        return True
def DBSCAN(data, eps, minPts):
    """
    输入：数据集, 半径大小, 最小点个数
    输出：分类簇id
    """
    clusterId = 1
    nPoints = data.shape[1]
    print("points: {}".format(nPoints))
    clusterResult = [UNCLASSIFIED] * nPoints
    for pointId in range(nPoints):
        point = data[:, pointId]
        # 判断是否已经完成分类
        if clusterResult[pointId] == UNCLASSIFIED:
            if expand_cluster(data, clusterResult, pointId, clusterId, eps, minPts):
                clusterId = clusterId + 1
    return clusterResult, clusterId - 1
 
# 使用 DBSCAN进行分类
dataset = array1.transpose()
clusters, clusterNum = DBSCAN(dataset, 15, 8)
print("clusterResult:{} cluster Numbers = {}".format(clusters,clusterNum))
from sklearn.metrics import davies_bouldin_score

davies_bouldin_score(array1, clusters)
from numpy import *
# 计算欧几里得距离
def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())

# 初始化 聚类中心
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = mat(zeros((k,n)))   # 每个聚类中心有n个坐标值，总共要k个中心
    for j in range(n):
        minJ = min(dataSet[:,j])
        maxJ = max(dataSet[:,j])
        rangeJ = float(maxJ - minJ)
        centroids[:,j] = minJ + rangeJ * random.rand(k, 1)
    return centroids
# k-means 聚类算法
def kMeans(dataSet, k):
    m = np.shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))    # 用于存放该样本属于哪类及距离
    # clusterAssment第一列存放其所属的中心点，第二列是该数据点到中心点的距离
    centroids = randCent(dataSet, k)
    clusterChanged = True   # 用来判断前后中心是否一致，即聚类是否已经收敛
    while clusterChanged:
        clusterChanged = False;
        for i in range(m):  # 把每一个数据点划分到离它最近的中心点
            minDist = inf; minIndex = -1;
            for j in range(k):
                # 计算到聚类中心距离
                distJI = dist(centroids[j,:], dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j  # 如果第i个数据点到第j个中心点最近，则将i聚类为j
            if clusterAssment[i,0] != minIndex: 
                clusterChanged = True  # 如果分配发生变化，则需要继续迭代
            clusterAssment[i,:] = minIndex,minDist**2   # 并将第i个数据点的分配情况存入字典
        for cent in range(k):   # 重新计算中心点
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A == cent)[0]]   # 去第一列等于cent的所有列
            centroids[cent,:] = mean(ptsInClust, axis = 0)  # 算出这些数据的中心点
    return centroids, clusterAssment

# 用测试数据及测试kmeans算法
myCentroids,clustAssing = kMeans(dataset.transpose(),3)
print(clustAssing[:,0].transpose())
# 计算DBI
davies_bouldin_score(array1, clustAssing[:,0])
from sklearn.cluster import KMeans
y_pred_km = KMeans(n_clusters=3, random_state=9).fit_predict(array1)
y_pred_km
davies_bouldin_score(array1, y_pred_km)