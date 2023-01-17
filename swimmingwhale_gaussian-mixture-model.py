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
def init(X,K):
    m,n = X.shape
    
    a = np.full((K),1/K)
    mu = [X[i] for i in range(K)]
    sigma = np.zeros((n,n))
    for i in range(n):
        sigma[i,i] = 0.1
    
    Sigma = [sigma for i in range(K)]
    return a,mu,Sigma
def gaussian(x,mean,cov):
    dim = np.shape(cov)[0] #维度
    #之所以加入单位矩阵是为了防止行列式为0的情况
    covdet = np.linalg.det(cov+np.eye(dim)*0.01) #协方差矩阵的行列式
    covinv = np.linalg.inv(cov+np.eye(dim)*0.01) #协方差矩阵的逆
    xdiff = x - mean
    #概率密度
    prob = 1.0/np.power(2*np.pi,1.0*dim/2)/np.sqrt(np.abs(covdet))*np.exp(-1.0/2*np.dot(np.dot(xdiff,covinv),xdiff))
    return prob
def GMM(data,K,iter_num = 10):
    m,dim = data.shape
    # m*K 的矩阵,代表着每一个x属于K个高斯分布的概率
    Q = np.zeros((m,K))
    a,mu,sigma = init(data,K)

    plt.plot(data[:,0],data[:,1],'x')

    for _ in range(iter_num):

        # E_step
        for i in range(m):
            respons = [a[k]*gaussian(data[i],mu[k],sigma[k]) for k in range(K)]
            sumrespons = np.sum(respons)
            Q[i] = respons/sumrespons

        # M_step
        for k in range(K):
            nk = np.sum(Q[:,k])
            a[k] = 1.0*nk/m
            mu[k] = 1.0/nk * np.sum([Q[i][k]*data[i] for i in range(m)],axis=0)

            xdiffs = data - mu[k]
            sigma[k] = 1.0/nk * np.sum([Q[i][k]*np.dot(xdiffs[i].reshape(1,-1).T,xdiffs[i].reshape(1,-1)) for i in range(m)],axis=0)

        # 绘制图像
        for i in range(K):
            plt.plot(mu[i][0],mu[i][1],'o',c = 'r')
    return a,mu,sigma

# 聚类个数
K = 3
# 迭代次数
iter_num = 20

a,mu,sigma = GMM(data,K,iter_num)
import numpy as np
import matplotlib.pyplot as plt

num = 200
l = np.linspace(0,15,num)
X, Y =np.meshgrid(l, l)
plt.plot(data[:,0],data[:,1],'x')

plt_data = np.dstack((X,Y))
for i in range(K):
    Z = [[gaussian(plt_data[k][j],mu[i],sigma[i]) for j in range(num)] for k in range(num)]
    cs = plt.contour(X,Y,Z)
    plt.clabel(cs)