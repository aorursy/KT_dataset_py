import numpy as np 

import pandas as pd 

import sklearn.utils as skutils

import sklearn.model_selection as skmodelsel

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
df = pd.read_csv('../input/CylinderVolume.csv')   #Training Dataset

df = skutils.shuffle(df)

dfTrain, dfValid = skmodelsel.train_test_split(df, test_size=0.2)

dfTrain.head()
dfTrain.plot(x='radius',y='volume',kind='scatter')
dfTrain.plot(x='height',y='volume',kind='scatter')
#%matplotlib notebook

plt3D = plt.figure().gca(projection='3d')

plt3D.scatter(dfTrain['radius'], dfTrain['height'], dfTrain['volume'])

plt3D.set_xlabel('radius')

plt3D.set_ylabel('height')

plt3D.set_zlabel('volume')

plt.show()
def extractFeatures(df):

    df_Features=df.loc[:,['radius','height']]

    df_Label=df.loc[:,['volume']]

    X=df_Features.values

    Y=df_Label.values

    return X,Y
X,Y=extractFeatures(dfTrain)
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
degree=3

inputX=mapFeature(XNorm,degree)  
batchSize=len(Y)         #no of Examples

iterations = 10000

alpha = 1000000000

beta1=0.99

beta2=0.999

learningDecayRate=0.9998

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

    

    #weights=weights - (alpha/batchSize)*vDW     #Momentum   

    #weights=weights - (alpha/batchSize)*dW/np.sqrt(csDW+epsilon)     #RMSProp 

    weights=weights - (alpha/batchSize)*(vDWc/(np.sqrt(sDWc)+epsilon)) #Adam          

    

    

    #Compute Loss for Plotting

    newLoss=np.matmul( inputX,weights)-Y

    newLossSqr=np.multiply(newLoss,newLoss)

    lossList[k]=(1.0/(2.0*batchSize))* np.sum(newLossSqr)



print("{0:.15f}".format(lossList[iterations-1][0]))
plt.plot(lossList[9000:10000],color='r')

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
radius=189

height=177

pi=3.14159265358979   #Same as Excel Function correct upto 15 decimal places

Volume=pi*radius**2*height

print("{0:.10f}".format(Volume))
pY=predict([[radius,height]], weights,SMean,SDev,degree)  # Predict with bias feature added

print("{0:.10f}".format(pY[0][0]))