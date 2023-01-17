import numpy as np
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
def findCoreObject(X,epsilon,minpts):
    m,n = X.shape
    distance = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            distance[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            
    return np.where(np.sum(distance<epsilon,axis=1) > minpts)[0],distance

epsilon = 0.5
minpts = 4

CoreObject,distance = findCoreObject(data,epsilon,minpts)
core_points = data[CoreObject]
plt.plot(core_points[:,0],core_points[:,1],'x',c='r')

D = np.arange(len(data))
otherObject = np.setdiff1d(D,CoreObject)
other_points = data[otherObject]
plt.plot(other_points[:,0],other_points[:,1],'x')


Corelist = CoreObject

k = 0

T = D
result = []

# 对所有核心对象进行处理
while len(Corelist) > 0 :
    Old_T = T
    # 建立队列,存放与该核心对象密度相连的所有对象
    Q = [Corelist[0]]
    # 将已访问数据弹出未访问集合
    T = np.setdiff1d(T,Corelist[0])
    # 处理队列中的每一个元素
    while len(Q) >0 :
        q = int(Q.pop())
        # 如果对象是核心对象
        if np.intersect1d(CoreObject,q).size > 0 :
            Nq = np.where(distance[q] < epsilon)[0]
            Delta = np.intersect1d(Nq,T)
            # 将未处理过的邻域中的对象放到Q队列,等待处理
            Q = np.union1d(Q,Delta).tolist()
            # 将已访问数据弹出未访问集合
            T = np.setdiff1d(T,Delta)
    k += 1
    C = np.setdiff1d(Old_T,T)
    result.append(C)
    Corelist = np.setdiff1d(Corelist,C)

# 绘制聚类结果
for i in result:
    points = data[i]
    plt.plot(points[:,0],points[:,1],'x')
