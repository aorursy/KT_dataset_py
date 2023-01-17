import numpy as np
import time
import matplotlib.pyplot as plt

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
def calcDistance(X):
    m,n = X.shape
    distance = np.zeros((m,m))
    for i in range(m):
        # 自己的距离设为最大值
        distance[i,i] = float('inf')
        for j in range(i+1,m):
            distance[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            distance[j,i] = distance[i,j]
    return distance


def calcClassAvgDistance(data,X,Y):
    x_center = np.sum(data[X],axis=0)/len(X)
    y_center = np.sum(data[Y],axis=0)/len(Y)
    return np.linalg.norm(x_center-y_center)
distance = calcDistance(data)
m,n = data.shape
C = np.arange(m).reshape(-1,1).tolist()
q = m
k = 3


while q > k :
    # 找最近的两个集合,合并
    i,j = np.unravel_index(np.argmin(distance, axis=None), distance.shape)
    C[i] = np.union1d(C[i],C[j]).tolist()
    # 因为j元素并入i的集合了,删除j元素原理的集合
    C = np.delete(C, j,0)
    # 因为j元素并入i的集合了,删除j元素在distance中的信息
    distance = np.delete(distance, j,0)
    distance = np.delete(distance, j,1)
    # 剩余类别数-1
    q = q -1
    # 重新计算新合并的元素与其他元素的距离
    for j in range(q):
        distance[i,j] = calcClassAvgDistance(data,C[i],C[j])
        distance[j,i] = distance[i,j]
    distance[i,i] = float('inf')
    if q < k+4:
        for i in C:
            points = data[i]
            plt.plot(points[:,0],points[:,1],'x')
        plt.show()

