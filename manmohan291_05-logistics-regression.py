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

            

X=np.array([X1,X2]).T

Y=np.array([Y1]).T
cmap = ListedColormap(['blue', 'red'])                    

plt.scatter(X1,X2, c=Y1,marker='.', cmap=cmap)

plt.show()
SMean=np.min(X,axis=0)    #using Min-Max Normalization

SDev=np.max(X,axis=0)

def NormalizeInput(X,SMean,SDev):   

    XNorm=(X-SMean)/SDev

    return XNorm
XNorm=NormalizeInput(X,SMean,SDev)
def mapFeature(X,degree):

    

    sz=X.shape[1]

    if (sz==2):

        sz=(degree+1)*(degree+2)/2

        sz=int(sz)

    else:

         sz=degree+1

    out=np.ones((X.shape[0],sz))     #Adding Bias W0



    sz=X.shape[1]

    if (sz==2):

        X1=X[:, 0:1]

        X2=X[:, 1:2]

        col=1

        for i in range(1,degree+1):        

            for j in range(0,i+1):

                out[:,col:col+1]= np.multiply(np.power(X1,i-j),np.power(X2,j))    

                col+=1

        return out

    else:

        for i in range(1,degree+1):        

            out[:,i:i+1]= np.power(X,i)

    

    return out
degree=2

inputX=mapFeature(XNorm,degree) 
def sigmoid(z):

    return 1/(1 + np.exp(-z))
def computeCost(weights,X,Y):

    n = X.shape[0]

    fx=np.matmul( X,weights)                      #Hypothesis

    hx=sigmoid(fx)

    term1=np.sum(np.multiply(Y,np.log(hx)))

    term2=np.sum(np.multiply(np.subtract(1,Y),np.log(1-hx)))    

    J=(-1/n)*(term1+term2)

    return J
batchSize=len(Y)         #no of Examples

iterations = 10000

alpha = 0.9

beta1=0.99

beta2=0.999

learningDecayRate=0.999998

epsilon=0.0000000001

featureCount=inputX.shape[1] 

weights=np.zeros((featureCount, 1)) #initialize Weight Paramters

vDW=np.zeros((featureCount, 1))

sDW=np.zeros((featureCount, 1))

lossList=np.zeros((iterations,1),dtype=float)  #for plotting loss curve


for k in range(iterations):

    #nth iteration

    t=k+1

    

    #Hypothesis

    fx=np.matmul( inputX,weights)           

    

    hx=sigmoid(fx)

    

    #Loss

    loss=hx-Y  

    

    

    #derivative

    dW=np.matmul(inputX.T,loss)  #Derivative

   

    #learning Rate decrease as training progresses 

    alpha=alpha*learningDecayRate

    

    #Moment Update

    vDW = (beta1) *vDW+ (1-beta1) *dW        #Momentum  

    sDW = (beta2) *sDW+ (1-beta2) *(dW**2)   #RMSProp

    

    #Bias Correction

    vDWc =vDW/(1-beta1**t)       

    sDWc =sDW/(1-beta2**t)

    

    #gradient Update

    #weights=weights - (alpha/batchSize)*dW                           #Simple

    weights=weights - (alpha/batchSize)*vDW                          #Momentum   

    #weights=weights - (alpha/batchSize)*dW/np.sqrt(csDW+epsilon)     #RMSProp 

    #weights=weights - (alpha/batchSize)*(vDWc/(np.sqrt(sDWc)+epsilon)) #Adam          

    

    

    #Compute Loss for Plotting

    lossList[k]=computeCost(weights,inputX,Y)



print("{0:.15f}".format(lossList[iterations-1][0]))

plt.plot(lossList,color='r')

plt.show
def predict(X,weights,SMean,SDev,degree):

    XNorm=NormalizeInput(X,SMean,SDev)

    inputX=mapFeature(XNorm,degree)

    fx=np.matmul(inputX, weights)

    hx=sigmoid(fx)

    PY=np.round(hx) 

    return PY

def accurracy(Y1,Y2):

    m=np.mean(np.where(Y1==Y2,1,0))    

    return m*100
pY=predict(X, weights,SMean,SDev,degree) 

print(accurracy(Y, pY))
plt.figure(figsize=(12,8))

plt.scatter(X[:,0],X[:,1], c=Y[:,0], cmap=cmap) 

###########################################################################

#Predict for each X1 and X2 in Grid 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

u = np.linspace(x_min, x_max, 50) 

v = np.linspace(y_min, y_max, 50) 



U,V=np.meshgrid(u,v)

UV=np.column_stack((U.flatten(),V.flatten())) 

W=predict(UV, weights,SMean,SDev,degree) 

plt.scatter(U.flatten(), V.flatten(),  c=W.flatten(), cmap=cmap,marker='.', alpha=0.1)



###########################################################################

#Exact Decision Boundry can be plot with contour

z = np.zeros(( len(u), len(v) )) 

for i in range(len(u)): 

    for j in range(len(v)): 

        uv= np.column_stack((np.array([[u[i]]]),np.array([[v[j]]])))               

        z[i,j] =predict(uv, weights,SMean,SDev,degree) 

z = np.transpose(z) 

plt.contour(u, v, z)

###########################################################################

plt.show()
XNorm=NormalizeInput(X,SMean,SDev)

inputX=mapFeature(XNorm,degree)

fx=np.matmul(inputX, weights)

hx=sigmoid(fx)

plt.figure(figsize=(12,8))

plt.scatter(fx,hx,c=np.round(hx), cmap=cmap)



x = np.arange(-18, 18, 0.1)

g = sigmoid(x)

plt.plot(x, g,color='g' ,linewidth=2,alpha=1)

plt.plot(x, x*0,color='k',linewidth=1,alpha=0.2)

plt.plot([-2,0,2], [0.5,0.5,0.5],color='r',alpha=0.8)

plt.plot([0,0], [-0.1,1],color='k',linewidth=1,alpha=0.2)

plt.xlabel('x')

plt.ylabel('$\sigma(z)$')

plt.show()