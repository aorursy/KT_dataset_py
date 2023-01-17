import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.stats import norm

from scipy.stats import multivariate_normal

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colors import ListedColormap

import itertools
X1=[]

X2=[]

Y1=[]

for i,j in itertools.product(range(50),range(50)):

    if abs(i-25)>5 and abs(i-j)<20 and np.random.randint(5,size=1) >0:

        X1=X1+[i/2]

        X2=X2+[j/2]

        if (i>25):

            Y1=Y1+[1]

        else:

            Y1=Y1+[0]

X=np.array([X1,X2]).T

Y=np.array([Y1]).T
cmap = ListedColormap(['blue', 'red'])                    

plt.scatter(X1,X2, c=Y1,marker='.', cmap=cmap)

plt.show()
def NaiveBayesClassifier(X,Y,Xtest):



    M0=np.mean(X[np.where(Y==0)[0]],axis=0)

    M1=np.mean(X[np.where(Y==1)[0]],axis=0)

    S0=np.std(X[np.where(Y==0)[0]],axis=0,ddof=1)

    S1=np.std(X[np.where(Y==1)[0]],axis=0,ddof=1)

    Ytest=np.zeros((Xtest.shape[0],1))

  

    for i in range(len(Xtest[:,0:1])): 

        Prob_X0_Y0= norm.pdf(Xtest[i,0],loc=M0[0], scale=S0[0])

        Prob_X0_Y1= norm.pdf(Xtest[i,0],loc=M1[0], scale=S1[0])

        Prob_X1_Y0= norm.pdf(Xtest[i,1],loc=M0[1], scale=S0[1])

        Prob_X1_Y1= norm.pdf(Xtest[i,1],loc=M1[1], scale=S1[1])

        Prob_Y0_X1X2 =(Prob_X0_Y0 *Prob_X1_Y0)/((Prob_X0_Y0 *Prob_X1_Y0)+(Prob_X0_Y1 *Prob_X1_Y1))

        Prob_Y1_X1X2 =(Prob_X0_Y1 *Prob_X1_Y1)/((Prob_X0_Y0 *Prob_X1_Y0)+(Prob_X0_Y1 *Prob_X1_Y1))

        if (Prob_Y1_X1X2>Prob_Y0_X1X2):

            Ytest[i]=1

        

    return Ytest
def accurracy(Y1,Y2):

    m=np.mean(np.where(Y1==Y2,1,0))    

    return m*100
K=25

pY=NaiveBayesClassifier(X,Y,X) 

print(accurracy(Y, pY))




#Predict for each X1 and X2 in Grid 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

u = np.linspace(x_min, x_max, 50) 

v = np.linspace(y_min, y_max, 50) 



U,V=np.meshgrid(u,v)

UV=np.column_stack((U.flatten(),V.flatten())) 

W=NaiveBayesClassifier(X,Y,UV) 

W.shape=U.shape

plt.contourf(U, V, W, cmap=cmap, alpha=0.2)



###########################################################################

plt.scatter(X[:,0],X[:,1], c=Y[:,0],marker="." ,cmap=cmap) 

###########################################################################

plt.show()
def plotNormalSurface(X,y):   

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')  

  

    M0=np.mean(X[np.where(y==0)[0]],axis=0)

    M1=np.mean(X[np.where(y==1)[0]],axis=0)

    S0=np.std(X[np.where(y==0)[0]],axis=0,ddof=1)

    S1=np.std(X[np.where(y==1)[0]],axis=0,ddof=1)

    V0=np.var(X[np.where(y==0)[0]],axis=0,ddof=1)

    V1=np.var(X[np.where(y==1)[0]],axis=0,ddof=1)

    

    x_min= M0[0]-4*S0[0]

    x_max =M0[0]+4*S0[0]

    y_min = M0[1]-4*S0[1]

    y_max = M0[1]+4*S0[1]

    u = np.linspace(x_min, x_max,50) 

    v = np.linspace(y_min, y_max,50) 

    



    U, V = np.meshgrid(u,v)

    pos = np.empty(U.shape + (2,))

    pos[:, :, 0] = U; pos[:, :, 1] = V



    rv = multivariate_normal([M0[0], M0[1]], [[V0[0], 0], [0, V0[1]]])

    W=rv.pdf(pos)



    ax.plot_surface(U,V,W,alpha=0.5, cmap='viridis',linewidth=0)







    rv = multivariate_normal([M1[0], M1[1]], [[V1[0], 0], [0, V1[1]]])

    W=rv.pdf(pos)



    ax.plot_surface(U,V,W,alpha=0.5,cmap='viridis',linewidth=0)



    ax.set_xlabel('X axis')

    ax.set_ylabel('Y axis')

    ax.set_zlabel('Z axis')

    

    

    

    plt.show()







    return



plotNormalSurface(X,Y)