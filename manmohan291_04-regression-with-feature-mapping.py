import numpy as np 

import pandas as pd 

import sklearn.utils as skutils

import sklearn.model_selection as skmodelsel

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/HousePriceData.csv')   #Training Dataset

df = skutils.shuffle(df)

dfTrain, dfValid = skmodelsel.train_test_split(df, test_size=0.2)

dfTrain.head()
pd.plotting.scatter_matrix(dfTrain, alpha=1, diagonal='kde',color='r')

plt.show()
#%matplotlib notebook

plt3D = plt.figure().gca(projection='3d')

plt3D.scatter(dfTrain['FloorArea'], dfTrain['BedRooms'], dfTrain['Price'],color="r")

plt3D.set_xlabel('FloorArea')

plt3D.set_ylabel('BedRooms')

plt3D.set_zlabel('Price')

plt.show()
def extractFeatures(df):

    df_Features=df.loc[:,['FloorArea','BedRooms']]

    df_Label=df.loc[:,['Price']]

    X=df_Features.values

    Y=df_Label.values

    return X,Y
X,Y=extractFeatures(dfTrain)
SMean=np.mean(X,axis=0)   

SDev=np.std(X,axis=0)

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
degree=3

inputX=mapFeature(XNorm,degree)  
batchSize=len(Y)         #no of Examples

iterations = 5000

alpha =100000000

beta1=0.9

beta2=0.999

learningDecayRate=0.998

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

    hypothesis=np.matmul( inputX,weights)           

    

    #Loss

    loss=hypothesis-Y  

    

    

    #derivative

    dW=np.matmul(inputX.T,loss)  #Derivative

   

    #learning Rate decrease as training progresses 

    alpha=alpha*learningDecayRate

    

    #gradient Update

    vDW = (beta1) *vDW+ (1-beta1) *dW        #Momentum  

    sDW = (beta2) *sDW+ (1-beta2) *(dW**2)   #RMSProp

    

    vDWc =vDW/(1-beta1**t)

    sDWc =sDW/(1-beta2**t)

    #weights=weights - (alpha/batchSize)*dW       #normal

    #weights=weights - (alpha/batchSize)*vDW     #Momentum   

    #weights=weights - (alpha/batchSize)*dW/np.sqrt(csDW+epsilon)     #RMSProp 

    weights=weights - (alpha/batchSize)*(vDWc/(np.sqrt(sDWc)+epsilon)) #Adam          

    

    

    #Compute Loss for Plotting

    newLoss=np.matmul( inputX,weights)-Y

    newLossSqr=np.multiply(newLoss,newLoss)

    lossList[k]=(1.0/(2.0*batchSize))* np.sum(newLossSqr)



print("{0:.15f}".format(lossList[iterations-1][0]))
plt.subplot(111)

plt.plot(lossList,color='r')

plt.xlabel('Full Plot')

plt.show()



plt.subplot(121)

plt.plot(lossList[0:500],color='b')

plt.xlabel('First 500 Values')





plt.subplot(122)

plt.plot(lossList[len(lossList)-500:len(lossList)],color='b')

plt.xlabel('Last 500 Values')

plt.show()
def predict(X,weights,SMean,SDev,degree):

    XNorm=NormalizeInput(X,SMean,SDev)

    inputX=mapFeature(XNorm,degree)

    PY=np.matmul(inputX, weights)

    return PY

def getRMSE(aY,pY):

    Error=aY- pY

    ErrorSqr=Error**2

    MSE=ErrorSqr.mean()

    RMSE=np.sqrt(MSE)

    return RMSE
pY=predict(X, weights,SMean,SDev,degree)  # Predict with bias feature added

print("{0:.15f}".format(getRMSE(Y, pY)))
vX,vY=extractFeatures(dfValid)

pY=predict(vX, weights,SMean,SDev,degree)  # Predict with bias feature added

print("{0:.15f}".format(getRMSE(vY, pY)))
pY=predict(X, weights,SMean,SDev,degree)  # Predict with bias feature added

plt.scatter(X[:,0],Y,color="r")

plt.scatter(X[:,0],pY[:,0],color="b")

plt.xlabel("FloorArea")

plt.ylabel("Price")

plt.legend(["Actual","Predicted"])

plt.show()
plt.close()

pY=predict(X, weights,SMean,SDev,degree)  # Predict with bias feature added

plt.scatter(X[:,1],pY[:,0],color="b")

plt.scatter(X[:,1],Y,color="r")

plt.xlabel("BedRooms")

plt.ylabel("Price")

plt.legend(["Actual","Predicted"])

plt.show()
%matplotlib notebook

fig = plt.figure()

plt3D = fig.add_subplot(111, projection='3d')   

plt3D.scatter(X[:,0],X[:,1],Y,marker="o",color="r")



x_min, x_max = X[:, 0].min() , X[:, 0].max() 

y_min, y_max = X[:, 1].min() , X[:, 1].max() 

u = np.linspace(x_min, x_max,20) 

v = np.linspace(y_min, y_max, 20) 

z = np.zeros(( len(u), len(v) )) 

U,V=np.meshgrid(u,v)

for i in range(len(u)): 

    for j in range(len(v)): 

        uv= np.column_stack((np.array([[u[i]]]),np.array([[v[j]]])))

        pv =predict(uv, weights,SMean,SDev,degree)

        z[i,j] =pv[0][0]

z = np.transpose(z) 

plt3D.plot_surface(U,V,z,alpha=0.5,color='b')

plt.show()