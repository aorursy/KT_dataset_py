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

plt.scatter(X1,X2, c=Y1, cmap=cmap,marker=".")

plt.show()
def svm_computeCost(X,Y,weights,C):

    n = X.shape[0]

    fx=np.matmul( X,weights)                      #Hypothesis

    hx=Y*fx  # no activation

    

    hx=1.0-hx

    hx[hx[:,0]<=0.] =[0.]

    

    term1=(1/n)*np.sum(hx)*C

    term2=(1/2)*(weights[1,0]**2+weights[2,0]**2)   

    J=term1+term2

    

    return J
def svm_gredients(X,Y,weights,C):

    n = X.shape[0]

    #Hypothesis

    fx=np.matmul(X,weights)  

    hx=Y*fx  # no activation

    hx=1-hx

    hx[hx[:,0]<=0.] =[0.]

    Loss=-Y*X*C/n

    Loss[hx[:,0]==0.] =[0.,0.,0.]

    loss=np.sum(Loss,axis=0)

    loss.shape=weights.shape

   

    #derivative

    dW=loss+weights*[[0.],[1.],[1.]] #Derivative

    

    return dW
def svm_train(X,Y,C,iterations=1000,alpha=0.1,plotLoss=False):

    

    

    inputX=np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #Add bias

    Y_New=np.array(Y)

    Y_New[Y<=0] =-1





    batchSize=len(Y_New)         #no of Examples



    #initialize Weight Paramters

    featureCount=inputX.shape[1] 

    weights=np.zeros((featureCount, 1)) 

    #weights=np.random.rand(featureCount, 1)*10



    #for plotting loss curve

    lossList=np.zeros((iterations,1),dtype=float)  

    



    for k in range(iterations):



        #derivative

        dW=svm_gredients(inputX,Y_New,weights,C)



        #gradient Update

        weights=weights - (alpha/batchSize)*dW 



        #Compute Loss for Plotting

        lossList[k]=svm_computeCost(inputX,Y_New,weights,C)

    if(plotLoss):

        print("{0:.15f}".format(lossList[len(lossList)-1][0]))

        plt.plot(lossList,color='r')

        plt.show()



    return weights



def accurracy(Y1,Y2):

    m=np.mean(np.where(Y1==Y2,1,0))    

    return m*100
def svm_predict(X,weights):

    inputX=np.concatenate((np.ones((X.shape[0],1)),X),axis=1) #Add bias

    hx=np.matmul(inputX, weights)

    hx[hx<=0.0] =0

    hx[hx>0.0] =1

    PY=np.round(hx) 

    return PY
def plot_Decision_Boundry(X,Y,weights):   

    plt.figure(figsize=(8,6))

    plt.scatter(X[:,0],X[:,1], c=Y[:,0],marker='.', cmap=cmap) 

 

    #Predict for each X1 and X2 in Grid 

    x_min, x_max = X[:, 0].min() , X[:, 0].max() 

    y_min, y_max = X[:, 1].min() , X[:, 1].max() 

    u = np.linspace(x_min, x_max, 100) 

    v = np.linspace(y_min, y_max, 100) 



    U,V=np.meshgrid(u,v)

    UV=np.column_stack((U.flatten(),V.flatten())) 

    W=svm_predict(UV, weights) 

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
X,Y=np.array([X1,X2]).T,np.array([Y1]).T

C=0.5



#train

weights=svm_train(X,Y,C,iterations=200000,plotLoss=True)



#Predict

pY=svm_predict(X, weights)

print("accurracy={:.2f} when C={:.2f} ".format(accurracy(Y, pY),C))



#Decision Boundry

plot_Decision_Boundry(X,Y,weights)
C=100



#train

weights=svm_train(X,Y,C,iterations=200000,plotLoss=True)



#Predict

pY=svm_predict(X, weights)

print("accurracy={:.2f} when C={:.2f} ".format(accurracy(Y, pY),C))



#Decision Boundry

plot_Decision_Boundry(X,Y,weights)