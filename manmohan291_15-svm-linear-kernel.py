import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

import itertools

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
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
def plot_Decision_Boundry(X,Y,clf):   

    

    w0=clf.intercept_

    w1=clf.coef_[0,0]

    w2=clf.coef_[0,1]

    weights=np.array([[w0,w1,w2]]).T



    plt.figure(figsize=(8,8))

    plt.scatter(X[:,0],X[:,1], c=Y[:,0],marker='.', cmap=cmap) 

 

    #Predict for each X1 and X2 in Grid 

    x_min, x_max = X[:, 0].min() , X[:, 0].max() 

    y_min, y_max = X[:, 1].min() , X[:, 1].max() 

    u = np.linspace(x_min, x_max, 100) 

    v = np.linspace(y_min, y_max, 100) 



    U,V=np.meshgrid(u,v)

    UV=np.column_stack((U.flatten(),V.flatten())) 

    W=clf.predict(UV) 

    plt.scatter(U.flatten(), V.flatten(),  c=W.flatten(), cmap=cmap,marker='.', alpha=0.1)

   

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


C=0.001

clf = SVC(kernel='linear',C=C)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))



plot_Decision_Boundry(X,Y,clf)



C=10

clf = SVC(kernel='linear',C=C)

clf.fit(X,Y.flatten())

y_pred = clf.predict(X)

print("Accuracy=",accuracy_score(Y.flatten(),y_pred))



plot_Decision_Boundry(X,Y,clf)