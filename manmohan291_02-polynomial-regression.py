import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

dfTrain = pd.read_csv('../input/Train.csv')   #Training Dataset

dfTest = pd.read_csv('../input/Test.csv')   #Test Dataset

dfValid = pd.read_csv('../input/Valid.csv') #Validation Dataset

dfTrain.head()
dfTrain.plot(x='X',y='Y',kind='scatter')
def extractFeatures(df):

    df_Features=df.iloc[:,0:1]

    df_Label=df.iloc[:,1:2]

    X=df_Features.values

    Y=df_Label.values

    return X,Y
X,Y=extractFeatures(dfTrain)
SMean=np.mean(X)

SDev=np.std(X)

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
degree=8

inputX=mapFeature(XNorm,degree)  
batchSize=len(Y)         #no of Examples

iterations = 1000

alpha = 0.01

featureCount=inputX.shape[1] 

weights=np.zeros((featureCount, 1)) #initialize Weight Paramters

lossList=np.zeros((iterations,1),dtype=float)  #for plotting loss curve


for k in range(iterations):

    #Hypothesis

    hypothesis=np.matmul( inputX,weights)           

    

    #Loss

    loss=hypothesis-Y  

    

    

    # derivative

    dW=np.matmul(inputX.T,loss)  #Derivative

   

    #gradient Update

    weights=weights - (alpha/batchSize)*dW              

    

    #Compute Loss for Plotting

    newLoss=np.matmul( inputX,weights)-Y

    newLossSqr=np.multiply(newLoss,newLoss)

    lossList[k]=(1.0/(2.0*batchSize))* np.sum(newLossSqr)

plt.plot(lossList,color='r')
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
X,Y=extractFeatures(dfTrain)

pY=predict(X, weights,SMean,SDev,degree)  # Predict with bias feature added

print(getRMSE(Y, pY))
vX,vY=extractFeatures(dfValid)

pY=predict(vX, weights,SMean,SDev,degree)  # Predict with bias feature added

print(getRMSE(vY, pY))
tX,tY=extractFeatures(dfTest)

pY=predict(tX, weights,SMean,SDev,degree)  # Predict with bias feature added

print(getRMSE(tY, pY))
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5

curveX = np.linspace(x_min, x_max, 100)

curveX.shape=(len(curveX),1) 

curveY=predict(curveX, weights,SMean,SDev,degree)  # Predict with bias feature added

plt.scatter(X,Y)

plt.plot(curveX, curveY,color='r')