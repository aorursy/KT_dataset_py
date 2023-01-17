import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import sklearn.linear_model  as LR

import sklearn.metrics as M

X=np.array([[-15.94,-29.15,36.19,37.49,-48.06,-8.94,15.31,

             -34.71,1.39,-44.38,7.01,22.76]]).T

Y=np.array([[2.13,1.17,34.36,36.84,2.81,2.12,14.71,

             2.61,3.74,3.73,7.63,22.75,]]).T

Xval=np.array([[-16.7,-14.6,34.5,-47.0,37.0,-40.7,-4.5,26.5,-42.8,25.4,-31.1,27.3

                ,-3.3,-1.8,-40.7,-50.0,-17.4,3.6,7.1,46.3,14.6]]).T

Yval=np.array([[4.2,4.1,31.9,10.6,31.8,5.0,4.5,22.3,-4.4,20.5,

                3.9,19.4,4.9,11.1,7.5,1.5,2.7,10.9,8.3,52.8,13.4]]).T

Xtest=np.array([[-33.3,-37.9,-51.2,-6.1,21.3,-40.3,-14.5,32.6,13.4,44.2,-1.1,

        -12.8,34.1,39.2,2.0,29.6,-23.7,-9.0,-55.9,-35.7,9.5]]).T

Ytest=np.array([[3.3,5.4,0.1,6.2,17.1,0.8,2.8,28.6,17.0,55.4,4.1,8.3,

        31.3,39.2,8.1,24.1,2.5,6.6,6.0,4.7,10.8]]).T
plt.scatter(X,Y)

plt.show()
def mapFeature(X,degree,includeBiasVector=True):

    

    sz=X.shape[1]

    if (sz==2):

        sz=(degree+1)*(degree+2)/2

        sz=int(sz)

    else:

         sz=degree+1



    out=np.ones((X.shape[0],sz))



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

    if (includeBiasVector==False):

        out=out[:,1:] #Remove Bias Vector



    return out
def SKLearnRegression(Xtrain, ytrain,degree,regAlpha):

    Xp=mapFeature(Xtrain,degree,False)    #Polynomial  

    if (regAlpha==0):

        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)

    else:

        RegObj=LR.Ridge(alpha=regAlpha,normalize=True).fit(Xp,ytrain)

    return RegObj

def SKLearnPredict(RegObj,X,degree):

    Xp=mapFeature(X,degree,False)    #Polynomial  

    Py=RegObj.predict(Xp)

    return Py
def SKLearnMSE(y_Actual,y_Predicted):

    MSE= M.mean_squared_error(y_Actual, y_Predicted)

    return MSE
def SKLearnPlotHypothesis(RegObj,X,y,degree,regAlpha):

    plt.scatter(X,y) 

    plt.title("Alpha="+str(regAlpha)+",Degree="+str(degree))

    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 

    u = np.linspace(x_min, x_max, 100)

    u.shape=(len(u),1) 

    v=SKLearnPredict(RegObj,u,degree) 

    plt.plot(u, v,color='r')

    return
plt.figure(figsize=(20,10))

regAlphaList=[0.0,0,0,0,0,0,0,1.0,3.0,5.0,10.0,100.0]      #Lambda is named as Alpha in Ridge Regression

degreeList=[1,2,3,4,5,6,8,8,8,8,8,8]

for i in range(len(regAlphaList)):

    regAlpha=regAlphaList[i]

    degree=degreeList[i]

    RegObj=SKLearnRegression(X,Y,degree,regAlpha)

    plt.subplot(2 , int(len(regAlphaList)/2 +0.5), i+1)

    SKLearnPlotHypothesis(RegObj,X,Y,degree,regAlpha)

plt.show()
def plotLearningCurve(Xtrain, ytrain, Xval, yval, degree,regAlpha):

    m = len(Xtrain)

    training_error = np.zeros((m, 1))

    validation_error   = np.zeros((m, 1))

    for i in range(m):

        Current_Xtrain=Xtrain[0:i+1]

        Current_ytrain=ytrain[:i+1]

        RegObj = SKLearnRegression(Current_Xtrain, Current_ytrain,degree,regAlpha) 

        predicted_ytrain=SKLearnPredict(RegObj,Current_Xtrain,degree)       

        training_error[i]=SKLearnMSE(Current_ytrain,predicted_ytrain)

        predicted_yval=SKLearnPredict(RegObj,Xval,degree)

        validation_error[i]=SKLearnMSE(yval,predicted_yval)

    

    plt.plot(range(1,m+1), training_error)

    plt.plot( range(1,m+1), validation_error)

    plt.title('Learning Curve (Alpha = '+str(regAlpha)+',Degree='+str(degree)+')')  

    plt.legend(('Training', 'Cross Validation'))   

    plt.xlabel("Training")

    plt.ylabel("MSE")

    return

plt.figure(figsize=(20,10))

regLambdaList=[0,0,0.01,1]

degreeList=[1,8,8,8]

for i in range(len(regLambdaList)):

    regLambda=regLambdaList[i]

    degree=degreeList[i]

    plt.subplot(2 , int(len(regLambdaList)/2 +0.5), i+1)

    plotLearningCurve(X,Y,Xval,Yval,degree,regLambda)

plt.show()
def plotValidationCurveForAlpha(Xtrain, ytrain, Xval, yval, degree,regAlphaList):

        

    training_error = np.zeros((len(regAlphaList), 1))

    validation_error   = np.zeros((len(regAlphaList), 1))



    for i in range(len(regAlphaList)):

        regAlpha=regAlphaList[i]

        RegObj = SKLearnRegression(Xtrain,ytrain,degree,regAlpha) 



        predicted_ytrain=SKLearnPredict(RegObj,Xtrain,degree)       

        training_error[i]=SKLearnMSE(ytrain,predicted_ytrain)

        

        predicted_yval=SKLearnPredict(RegObj,Xval,degree)

        validation_error[i]=SKLearnMSE(yval,predicted_yval)    

    plt.plot(regAlphaList, training_error)

    plt.plot( regAlphaList, validation_error)

    plt.title('Validation Curve (Degree='+str(degree)+')')  

    plt.legend(('Training', 'Cross Validation'))   

    plt.xlabel("Alpha")

    plt.ylabel("MSE")

    return
plt.figure(figsize=(20,10))

regAlphaList=[0,0,0.01,1]

degreeList=[1,8,8,8]

for i in range(len(regAlphaList)):

    regAlpha=regAlphaList[i]

    degree=degreeList[i]

    plt.subplot(2 , int(len(regAlphaList)/2 +0.5), i+1)

    plotLearningCurve(X,Y,Xval,Yval,degree,regAlpha)

plt.show()
def plotFinalCurve(Xtrain, ytrain, Xtest, ytest, degree,regAlpha):

    RegObj = SKLearnRegression(Xtrain,ytrain,degree,regAlpha)

    predicted_ytest=SKLearnPredict(RegObj,Xtest,degree)

    testErr=SKLearnMSE(ytest,predicted_ytest)

    #PLOT   

    X=np.concatenate((Xtrain,Xtest),axis=0)

    y=np.concatenate((ytrain,ytest),axis=0)

    x_min, x_max = X[:, 0].min()-1 , X[:, 0].max()+1 

    u = np.linspace(x_min, x_max, 100)

    u.shape=(len(u),1) 

    v=SKLearnPredict(RegObj,u,degree)

    plt.plot(u, v,color='r')

    plt.scatter(Xtrain,ytrain) 

    plt.scatter(Xtest,ytest)

    plt.title("Test data Alpha="+str(regAlpha ) +" , degree="+str(degree)+" with MSE="+str(round(testErr,4)))

    plt.legend(("Regression(Alpha=3,degree=8)","Training Data","Test Data"))

    return
plt.figure(figsize=(10,5))

degree=8

regLambda=3

plotFinalCurve(X,Y,Xtest,Ytest,degree,regLambda)

plt.show()