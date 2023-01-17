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
def addBiasFeature(X):

    inputX=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)

    return inputX
inputX=addBiasFeature(X)
batchSize=len(Y)         #no of Examples in batch

iterations = 5000

alpha = 0.001

lossList=np.zeros((iterations,1),dtype=float)  #for plotting loss curve

featureCount=inputX.shape[1]   #no of features + 1 (after added bias term)

weights=np.zeros((featureCount, 1)) #initialize Weight Paramters



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
def predict(X,weights):

    inputX=addBiasFeature(X)

    pY=np.matmul(inputX, weights)

    return pY
def getRMSE(aY,pY):

    Error=aY- pY

    ErrorSqr=Error**2

    MSE=ErrorSqr.mean()

    RMSE=np.sqrt(MSE)

    return RMSE
X,Y=extractFeatures(dfTrain)

pY=predict(X, weights)  # Predict with bias feature added

print(getRMSE(Y, pY))
vX,vY=extractFeatures(dfValid)

pY=predict(vX, weights)  # Predict with bias feature added

print(getRMSE(vY, pY))
tX,tY=extractFeatures(dfTest)

pY=predict(tX, weights)  # Predict with bias feature added

print(getRMSE(tY, pY))
x_min, x_max = X[:, 0].min() - 20, X[:, 0].max() + 20

lineX = np.linspace(x_min, x_max, 100)

lineX.shape=(len(lineX),1) 

lineY=predict(lineX, weights)  # Predict with bias feature added

plt.scatter(X,Y)

plt.plot(lineX, lineY,color='r')