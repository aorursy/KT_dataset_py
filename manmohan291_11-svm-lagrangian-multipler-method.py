import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import itertools
X1=[]

X2=[]

Y1=[]



for i,j in itertools.product(range(50),range(50)):

    if abs(i-j)>5 and abs(i-j)<40 and np.random.randint(5,size=1) >0:

        X1=X1+[i/2]

        X2=X2+[j/2]

        if (i>j):

            Y1=Y1+[1]

        else:

            Y1=Y1+[0]

### Add few misclassified Data            

for i,j in itertools.product(range(50),range(50)):

    if abs(i-j)<2  and abs(i-25)>13 and  np.random.randint(10,size=1) >7:        

        if (i<25):

            X1=X1+[i/2]

            X2=X2+[j/2-1.5]

            Y1=Y1+[0]

        else:

            X1=X1+[i/2]

            X2=X2+[j/2+1]

            Y1=Y1+[1]



X=np.array([X1,X2]).T

Y=np.array([Y1]).T
cmap = ListedColormap(['blue', 'red'])                    

plt.scatter(X1,X2, c=Y1,marker='.', cmap=cmap)

plt.show()
def SVM_Train(X, Y, C,  tol=0.001, max_passes= 50):

    m,n = X.shape

    y=np.array(Y)    # not to overwrite Y referenced here

    y[y==0] = -1

    alphas = np.zeros((m, 1))

    b = 0

    E = np.zeros((m, 1))

    passes = 0

    eta = 0

    L = 0

    H = 0

    y=y.flatten()

    E=E.flatten()

    alphas=alphas.flatten()

    

    K = np.matmul(X,X.T) # Linear Kernel

    

       

    while (passes < max_passes):  

        num_changed_alphas = 0

        for i in range(m):

            E[i] = b + np.sum(alphas*y*K[:,i]) - y[i]

            if ((y[i]*E[i] < -tol and alphas[i] < C) or (y[i]*E[i] > tol and alphas[i] > 0)):

                j= np.random.randint(0,m) 

                while (i==j):

                    j= np.random.randint(0,m) 

                E[j] = b + np.sum(alphas*y*K[:,j]) - y[j]

                alpha_i_old = alphas[i]

                alpha_j_old = alphas[j]

                if (y[i] == y[j]):

                    L = np.max([0, alphas[j] + alphas[i] - C])

                    H = np.min([C, alphas[j] + alphas[i]])

                else:

                    L =np.max([0, alphas[j] - alphas[i]])

                    H = np.min([C, C + alphas[j] - alphas[i]])

                if (L == H):

                    continue

                eta = 2.0 * K[i,j] - K[i,i] - K[j,j]

                if (eta >= 0): 

                    continue

                alphas[j] = alphas[j] -(y[j] * (E[i] - E[j])) / eta

                alphas[j] = np.min ([H, alphas[j]])

                alphas[j] = np.max ([L, alphas[j]])

                if (np.abs(alphas[j] - alpha_j_old) < tol):

                    alphas[j] = alpha_j_old

                    continue

                alphas[i] = alphas[i] + y[i]*y[j]*(alpha_j_old - alphas[j])

                b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - y[j] * (alphas[j] - alpha_j_old) *  K[i,j]

                b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - y[j] * (alphas[j] - alpha_j_old) *  K[j,j]

                if (0 < alphas[i] and alphas[i] < C):

                    b = b1

                elif (0 < alphas[j] and alphas[j] < C):

                    b = b2

                else:

                    b = (b1+b2)/2

                num_changed_alphas += 1

            #END IF

        #END FOR

        if (num_changed_alphas == 0):

            passes = passes + 1

        else:

            passes = 0

    #end while

    W=np.matmul((alphas*y).reshape(1,m),X).T

    weights=np.row_stack(([[b]],W))

    return weights

def predict(X,weights):

    fx=weights[0,0]+np.matmul(X, weights[1:,:])

    fx[fx>0]=1

    fx[fx<=0]=0

    PY=fx

    return PY
def accurracy(Y1,Y2):

    m=np.mean(np.where(Y1==Y2,1,0))    

    return m*100
def plot_Decision_Boundry(X,Y,weights):   

    plt.figure(figsize=(8,8))

    plt.scatter(X[:,0],X[:,1], c=Y[:,0],marker='.', cmap=cmap) 

 

    #Predict for each X1 and X2 in Grid 

    x_min, x_max = X[:, 0].min() , X[:, 0].max() 

    y_min, y_max = X[:, 1].min() , X[:, 1].max() 

    u = np.linspace(x_min, x_max, 100) 

    v = np.linspace(y_min, y_max, 100) 



    U,V=np.meshgrid(u,v)

    UV=np.column_stack((U.flatten(),V.flatten())) 

    W=predict(UV, weights) 

    plt.scatter(U.flatten(), V.flatten(),  c=W.flatten(), cmap=cmap,marker='.', alpha=0.1)

   



    # w1.x1+ w2.x2 +b=0

    # x2= -b/w2 - w1/w2*x1

    

    #Exact Decision Boundry 

    

    for i in range(len(u)): 

        v[i]=-weights[0,0]/weights[2,0]  -weights[1,0]* u[i]/weights[2,0]    

    plt.plot(u, v, color='k')

    ###########################################################################

    

    #Margins

    M= (2/np.sqrt(weights[1,0]**2+weights[2,0]**2))

  

   

    

    for i in range(len(u)): 

        v[i]=M-weights[0,0]/weights[2,0]  -weights[1,0]* u[i]/weights[2,0]    

    plt.plot(u, v, color='gray')

    

    for i in range(len(u)): 

        v[i]=-M-weights[0,0]/weights[2,0]  -weights[1,0]* u[i]/weights[2,0]    

    plt.plot(u, v, color='gray')

    

    

    plt.show()


C=0.01

weights=SVM_Train(X, Y, C=C)



pY=predict(X, weights) 

print("Accurracy=",accurracy(Y, pY))



plot_Decision_Boundry(X,Y,weights)
C=5

weights=SVM_Train(X, Y, C=C)



pY=predict(X, weights) 

print("Accurracy=",accurracy(Y, pY))



plot_Decision_Boundry(X,Y,weights)