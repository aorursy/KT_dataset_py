## This is a simple classification of  non linear data with  SVM(concentric circle dataset)
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
#reduce the value of noise to seperate the points more easily
X,Y=make_circles(n_samples=500,noise=0.02)
print(X.shape,Y.shape)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()
#we will project into higher dimension and now every example will have one more feature
def phi(X):
    """"Non Linear Transformation"""
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X1**2 + X2**2
    
    X_ = np.zeros((X.shape[0],3))
    print(X_.shape)
    
    X_[:,:-1] = X
    X_[:,-1] = X3

    return X_

X_=phi(X)
X_[:3,:]
def plot3d(X,show=True):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    X1 = X[:,0]
    X2 = X[:,1]
    X3 = X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=30,c=Y,depthshade=True)
    if show==True:
        plt.show()
    
    return ax
plot3d(X_)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
lr=LogisticRegression()
#calculating the cross val score
acc=cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f"%(acc*100))
acc=cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(3D) is %.4f"%(acc*100))
lr.fit(X_,Y)
#now to get the weights
wts=lr.coef_
wts
#to get the bias term
bias=lr.intercept_
bias
#now to find the value of z , you need to create some values for x any y
xx,yy=np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print(yy)

 #now generate the z matrix using xx and yy
z=-(wts[0,0]*xx+wts[0,1]*yy+bias)/wts[0,2]
z
#now plot the hyperplane
#we did show==true so that we can draw the hyper plane on the same graph and we can see them together.
#that is why we return ax too.
ax=plot3d(X_,False)
ax.plot_surface(xx,yy,z,alpha=0.2)
plt.show()



