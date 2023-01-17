import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv("../input/trainData3D.csv")
dx = data.X.values 
dy = data.Y.values
dz = data.Z.values
X = np.append(dx.reshape(-1,1),dy.reshape(-1,1), axis =1)
Y = dz
X.shape, Y.shape
ax = Axes3D(plt.figure())
ax.scatter(X[:,0],X[:,1],Y)
ax
def getW(X,q,tau):
    
    #Create W
    m = X.shape[0]
    W = np.eye(m) 
    
    for i in range(m):
        W[i,i] = np.exp(-np.dot((X[i]-q),(X[i]-q).T)/(2*tau*tau))
    
    return W
    
def getTheta(X,Y,q,tau):
    m = X.shape[0]
    ones = np.ones((m,1))
    q = np.append(np.array([1]), q, axis = 0)
    X = np.append(ones, X, axis = 1)
    W = getW(X,q,tau)
    Y = Y.reshape((-1,1))
    
    theta = np.dot(np.linalg.pinv(np.dot(np.dot(X.T,W),X)),np.dot(np.dot(X.T,W),Y))
    return theta,W
    
    
theta,W = getTheta(X,Y,[0.6,0.7],0.1)
print(theta.shape)
print(W)
# X_Test = np.linspace(-20,20,100)
X_Test = pd.read_csv("../input/testData3D.csv").values
# print(X_Test)
Y_Test = []

for xt in X_Test:
#     print(xt)
    theta,W = getTheta(X,Y,xt,0.73)
#     print(xt)
    pred = theta[0][0]*1 + theta[1][0]*xt[0] + theta[2][0]*xt[1]
    Y_Test.append(pred)
    
Y_Test = np.array(Y_Test)
Y_actual = pd.read_csv("../input/actualYTest3D.csv").values
Y_Test.shape, Y_actual.shape
from sklearn.metrics import r2_score
r2_score(Y_actual,Y_Test)


ax = Axes3D(plt.figure())
ax.scatter(
    X_Test[:,0],
    X_Test[:,1],
    Y_actual.reshape(-1,1)
)
plt.title("Redrawn predictions!")

X_Test[:,1].shape,X_Test[:,0].shape, Y.shape
from sklearn.model_selection import train_test_split as tts
xtrain,xtest,ytrain,ytest = tts(X,Y, random_state = 1)

from sklearn.linear_model import LinearRegression as LR
reg = LR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)
from sklearn.svm import SVR
reg = SVR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)
from sklearn.ensemble import GradientBoostingRegressor as GBR
reg = GBR()
reg.fit(xtrain,ytrain)
reg.score(xtest,ytest)
