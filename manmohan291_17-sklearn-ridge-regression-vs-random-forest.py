import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import sklearn.linear_model  as LR

import sklearn.svm as SM

import sklearn.ensemble as RF

import sklearn.metrics as M
X=np.array([[-15.94,-29.15,36.19,37.49,-48.06,-8.94,15.31,

             -34.71,1.39,-44.38,7.01,22.76]]).T

Y=np.array([[2.13,1.17,34.36,36.84,2.81,2.12,14.71,

             2.61,3.74,3.73,7.63,22.75,]]).T

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
def SKLearnRegression(Xtrain, ytrain,degree,regAlpha,algorithm):

    Xp=mapFeature(Xtrain,degree,False)    #Polynomial  

    if (algorithm=="Linear"):

        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)

    elif (algorithm=="Ridge"):

        RegObj=LR.Ridge(alpha=regAlpha,normalize=True).fit(Xp,ytrain)

    elif (algorithm=="SVR"):

        RegObj=SM.SVR(degree=degree).fit(Xp,ytrain)

    elif (algorithm=="RandomForest"):

        RegObj=RF.RandomForestRegressor().fit(Xp,ytrain)

    else:

        RegObj=LR.LinearRegression(normalize=True).fit(Xp,ytrain)

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

regAlpha=None

degree=2

RegObj=SKLearnRegression(X,Y,degree,regAlpha,"Linear")

plt.subplot(231)

SKLearnPlotHypothesis(RegObj,X,Y,degree,regAlpha)

predicted_y=SKLearnPredict(RegObj,X,degree)      

MSE1=SKLearnMSE(Y,predicted_y)

plt.title("Linear Regression")

plt.ylabel("MSE="+str(round(MSE1,3)) )

plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))

plt.show()
plt.figure(figsize=(20,10))

regAlpha=1

degree=8

RegObj=SKLearnRegression(X,Y,degree,regAlpha,"Ridge")

plt.subplot(232)

SKLearnPlotHypothesis(RegObj,X,Y,degree,regAlpha)

predicted_y=SKLearnPredict(RegObj,X,degree)      

MSE2=SKLearnMSE(Y,predicted_y)

plt.title("Ridge Regression")

plt.ylabel("MSE="+str(round(MSE2,3)))

plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))

plt.show()
plt.figure(figsize=(20,10))

regAlpha=None

degree=8

RegObj=SKLearnRegression(X,Y.flatten(),degree,regAlpha,"RandomForest")

plt.subplot(235)

SKLearnPlotHypothesis(RegObj,X,Y,degree,regAlpha)

predicted_y=SKLearnPredict(RegObj,X,degree)      

MSE4=SKLearnMSE(Y,predicted_y)

plt.xlabel("Random Forest Regression") 

plt.ylabel("MSE="+str(round(MSE4,3))) 

plt.legend(("Degree="+str(degree)," Alpha="+str(regAlpha)))

plt.show()
plt.figure(figsize=(10,5))

AlgoNames = ('Linear', 'Ridge',  'RandomForest')

AlgoIndex = [1,2,3]

AlgoMSE = [MSE1,MSE2,MSE4]

plt.bar(AlgoIndex, AlgoMSE, align='center', alpha=0.5)

plt.xticks(AlgoIndex, AlgoNames)

plt.ylabel('MSE')

plt.title('Algorithm Mean Squared Error')

plt.show()
