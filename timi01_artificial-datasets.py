import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
x1 = np.random.randn(1000)*1.5-4
x2 = np.random.randn(1000)*1.5

x3 = np.random.randn(1000)*0.5+4
x4 = np.random.randn(1000)*0.5
X1 = np.hstack([x1,x3])
X2 = np.hstack([x2,x4])

X = np.vstack([X1,X2]).T
y = np.ones(2000)
y[1000:]=2
plt.scatter(X[:,0],X[:,1],c=y, cmap='rainbow')
plt.axis('equal');
theta = np.linspace(0,2*np.pi,1000)
x1 = np.random.randn(1000)*0.1+np.sin(theta)
y1 = np.random.randn(1000)*0.1+np.cos(theta)

x2 = np.random.randn(1000)*0.2
y2 = np.random.randn(1000)*0.2
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.axis('equal');
x1 = np.random.randn(1000)
x2 = np.random.randn(1000)

x3 = np.random.randn(1000)*0.2
x4 = np.random.randn(1000)*0.2

r = np.sqrt(x1**2+x2**2)
r = (r<2.5) & (r>1.5)

x1 = x1*r
x2 = x2*r

X = np.concatenate([[x1,x2],[x3,x4]], axis=1).T
Y = np.hstack([np.zeros(1000), np.ones(1000)])
plt.scatter(X[:,0], X[:,1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.axis('equal');
x_l = np.random.rand(1000)*0.1
y_l = np.random.rand(1000)+1.1

x_r = np.random.rand(1000)*0.1+0.9
y_r = np.random.rand(1000)+1.1

x_b = np.random.rand(1000)
y_b = np.random.rand(1000)*0.1+1

x_t = np.random.rand(1000)
y_t = np.random.rand(1000)*0.1+2

x_m = np.random.randn(10000)*0.07+0.5
y_m = np.random.randn(10000)*0.07+1.5
# left
plt.scatter(x_l,y_l)

# right
plt.scatter(x_r,y_r)

# botton
plt.scatter(x_b,y_b)

# top
plt.scatter(x_t,y_t)

# middle
plt.scatter(x_m, y_m)
plt.axis('equal');
x1 = np.random.randn(1000)
y1 = np.random.randn(1000)

x2 = np.random.randn(1000)*0.2
y2 = np.random.randn(1000)*0.2
r = np.sqrt(x1**2+y1**2)>1.5
plt.scatter(x1*r,y1*r)
plt.scatter(x2,y2)
plt.axis('equal');
mean = [0,1]
var = [[1,-0.5],[-0.5,1]]

x , y = np.random.multivariate_normal(mean, var, 1000).T 

mean1 = [3,1]
var1 = [[1,-0.5],[-0.5,1]]

x1 , y1 = np.random.multivariate_normal(mean1, var1, 1000).T 
plt.scatter(x,y)
plt.scatter(x1,y1);
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# let's visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='winter');
from sklearn.datasets import make_gaussian_quantiles
X, y = make_gaussian_quantiles(mean=None, cov=0.5, n_samples=200, n_features=2, n_classes=2, shuffle=True, random_state=None)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
from sklearn.datasets import make_blobs
X, y = make_blobs(200, random_state=5, n_features=2, centers=6)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
from sklearn.datasets import make_circles
X, y = make_circles(200, noise=.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
from sklearn.datasets import make_moons
X, y = make_moons(200, noise=.05, random_state=0)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='winter');
m = 400 # number of examples
N = int(m/2) # number of points per class
D = 2 # dimensionality
X = np.zeros((m,D)) # data matrix where each row is a single example
Y = np.zeros((m), dtype='uint8') # labels vector (0 for red, 1 for blue)
a = 4 # maximum ray of the flower

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    Y[ix] = j
        
X = X.T
y = Y.T
plt.scatter(X[0,:], X[1,:], c=Y, s=40, cmap='winter');
X_c=np.random.rand(2000)
Y_c=np.random.rand(2000)
c1=X_c*0.6
c1=Y_c*0.6
X_c=X_c-c1
Y_c=Y_c-c1
X_c1=np.random.rand(2000)
Y_c1=np.random.rand(2000)
c11=X_c1*0.6 
c11=Y_c1*0.6 - 1
X_c1=X_c1-c11
Y_c1=Y_c1-c11
plt.scatter(X_c, Y_c, c='r')
plt.scatter(X_c1, Y_c1 -1 , c='b')
plt.axis('equal');
k = 200
n = np.random.rand(k,1) * 720 * (2*np.pi)/360
x1 = -np.cos(n)*n + np.random.rand(k,1) * 0.5
y1 = np.sin(n)*n + np.random.rand(k,1) * 0.5
X = np.vstack((np.hstack((x1,y1)),np.hstack((-x1,-y1))))
y = np.hstack((np.zeros(k),np.ones(k)))
X = np.vstack((np.hstack((x1,y1)),np.hstack((-x1,-y1))))
y = np.hstack((np.zeros(k),np.ones(k)))
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.axis('equal');