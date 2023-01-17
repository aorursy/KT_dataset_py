# import packages
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy

# show files
import os
print(os.listdir("../input"))
# set pyplot parameters to make things pretty
plt.rc('axes', linewidth = 1.5)
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('xtick.major', size = 3, width = 1.5)
plt.rc('ytick.major', size = 3, width = 1.5)
# read data
linearData = pd.read_csv('../input/linear.csv')
linearData.head()
# Let's first plot (x,y) and see what it looks like
plt.plot('x','y',data = linearData, marker = 'o', linestyle = '', label = 'data')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
x = linearData['x'].tolist()
y = linearData['y'].tolist()

# Don't forget - adding ones to the x matrix
xb = np.c_[np.ones((len(x),1)),x]
# calculate linear regression parameters theta using the normal equation
thetaHat = np.linalg.inv(xb.T.dot(xb)).dot(xb.T).dot(y)
print(thetaHat)
# thetaHat[0] is intercept, thetaHat[1] is slope. This is determined by the column order of matrix xb.
# plot the fit and the data
xFit = np.linspace(0,100,num = 200)
xFitb = np.c_[np.ones((len(xFit),1)), xFit]
yFit = xFitb.dot(thetaHat)

plt.plot('x','y',data = linearData, marker = 'o', linestyle = '', label = 'data')
plt.plot(xFit, yFit, color = 'r', lw = 3, linestyle = '--', label = 'Linear fit')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
# create the model
lin_reg = LinearRegression()
# format x so that LinearRegression recognize it.
x = np.array(x).reshape(-1,1)
# fit the model
lin_reg.fit(x,y)
lin_reg.intercept_, lin_reg.coef_
xb = sm.add_constant(x) # again, add a column of ones to x
model = sm.OLS(y,xb) # OLS = Ordinary Least Squares
results = model.fit()
print(results.summary())
learningRate = 0.0002
numIterations = 100000
y = np.array(y).reshape(-1,1)
m = len(y) # number of samples

# random initialization with standard normal distribution
theta = np.random.randn(2,1)

# start gradient descent
for i in range(numIterations):
    gradient = 2/m * xb.T.dot(xb.dot(theta) - y) # dimension: (2,1)
    theta = theta - learningRate * gradient
theta
# define the function to calculate MSE
# can also use sklearn.metrics.mean_squared_error
def MSE(xb,y,theta):
    return np.sum(np.square(xb.dot(theta)-y))/len(y)
learningRate = 0.0002
numIterations = 100000
y = np.array(y).reshape(-1,1)
m = len(y) # number of samples

# random initialization with standard normal distribution
theta = np.random.randn(2,1)

cost = []
# start gradient descent
for i in range(numIterations):
    gradient = 2/m * xb.T.dot(xb.dot(theta) - y) # dimension: (2,1)
    theta = theta - learningRate * gradient
    cost.append(MSE(xb,y,theta))
fig,ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
ax[0].plot(range(0,100),cost[0:100])
ax[1].plot(range(10000,20001), cost[10000:20001])
plt.subplots_adjust(wspace=0.5)
ax[0].set_xlabel('# of iterations', fontsize = 14)
ax[1].set_xlabel('# of iterations', fontsize = 14)
ax[0].set_ylabel('MSE', fontsize = 14)
ax[1].set_ylabel('MSE', fontsize = 14)
learningRate = 0.0003
numIterations = 100
y = np.array(y).reshape(-1,1)
m = len(y) # number of samples

# random initialization with standard normal distribution
theta = np.random.randn(2,1)

cost = []
# start gradient descent
for i in range(numIterations):
    gradient = 2/m * xb.T.dot(xb.dot(theta) - y) # dimension: (2,1)
    theta = theta - learningRate * gradient
    cost.append(MSE(xb,y,theta))

fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
ax.plot(range(0,100), cost[0:100])
ax.set_xlabel('# of iterations', fontsize = 14)
ax.set_ylabel('MSE', fontsize = 14)
theta = np.random.randn(2,1)
gradient = 2/m * xb.T.dot(xb.dot(theta) - y)
print(gradient)
xbStandard = copy.deepcopy(xb) # we don't want to mess with xb! xbStandard = xb will lead to xb being normalized, too.
# save the shift and scaling
mu = np.mean(xbStandard[:,1]) 
sigma = np.std(xbStandard[:,1])
# standardization
xbStandard[:,1]=(xbStandard[:,1]-mu)/sigma
print(xbStandard[0:5])
print(mu)
print(sigma)
learningRate = 0.1
numIterations = 1000

m = len(y) # number of samples

# random initialization with standard normal distribution
theta = np.random.randn(2,1)

cost = []
# start gradient descent
for i in range(numIterations):
    gradient = 2/m * xbStandard.T.dot(xbStandard.dot(theta) - y) # dimension: (2,1)
    theta = theta - learningRate * gradient
    cost.append(MSE(xbStandard,y,theta))

fig,ax = plt.subplots(nrows = 1, ncols = 1, figsize = (5,4))
plt.plot(cost)
ax.set_xlabel('# of iterations', fontsize = 14)
ax.set_ylabel('MSE', fontsize = 14)
plt.show()
print(theta)
xFit = np.linspace(0,100,num = 200)
xFitStandard = (xFit - mu)/sigma # use the previously saved mean and standard deviation
xFitStandardb = np.c_[np.ones((len(xFitStandard),1)), xFitStandard]
yFit = xFitStandardb.dot(theta)

plt.plot('x','y',data = linearData, marker = 'o', linestyle = '', label = 'data')
plt.plot(xFit, yFit, color = 'r', lw = 3, linestyle = '--', label = 'Linear fit')
plt.xlabel('x',fontsize = 18)
plt.ylabel('y', fontsize = 18)
plt.legend(fontsize = 14)
plt.show()
numEpochs = 1000

# a simple learning schedule
def learningSchedule(step):
    return 5/(50000 + step)

# visualize the learning schedule
s = np.linspace(0,15000, num = 15001)
l = learningSchedule(s)
m = len(y) # sample size 

fig, ax = plt.subplots(figsize = (8,4))
plt.plot(s,l, lw = 2)
#plt.xlim(0,15000)
plt.xlabel('step', fontsize = 18)
plt.ylabel('learning rate', fontsize = 18)
plt.show()
theta = np.random.randn(2,1)

cost = []
for epoch in range(numEpochs):
    for i in range(m):
        idx = np.random.randint(m) # 0,1,...,m-1
        xbi = xb[idx:idx+1]
        yi = y[idx:idx+1]
        gradient = 2*xbi.T.dot(xbi.dot(theta)-yi) # sample size is one
        learningRate = learningSchedule(epoch*m + i) # step = epoch*m + i
        theta = theta - learningRate * gradient
        cost.append(MSE(xb,y,theta))
print(theta)
fig, ax = plt.subplots(figsize = (8,4))
plt.plot(cost)
#plt.xlim(0,20000)
plt.ylabel('cost', fontsize = 18)
plt.xlabel('step', fontsize = 18)
plt.show()
numEpochs = 100

# a simple learning schedule
def learningSchedule(step):
    return 5/(10000 + step)

theta = np.random.randn(2,1)

cost = []
for epoch in range(numEpochs):
    for i in range(m):
        idx = np.random.randint(m) # 0,1,...,m-1
        xbi = xb[idx:idx+1]
        yi = y[idx:idx+1]
        gradient = 2*xbi.T.dot(xbi.dot(theta)-yi) # sample size is one
        learningRate = learningSchedule(epoch*m + i) # step = epoch*m + i
        theta = theta - learningRate * gradient
        cost.append(MSE(xb,y,theta))
plt.plot(cost)
plt.show()
from sklearn.linear_model import SGDRegressor
# max_iter is the total number of epochs, eta0 is the starting learning rate
# penalty = None, meaning there is no regularization.
model = SGDRegressor(eta0 = 0.0005, penalty = None, max_iter = 10000)
model.fit(x,y.ravel())
model.intercept_, model.coef_
from sklearn.utils import shuffle
batchsize = 30 # size of each of the mini batch
theta = np.random.randn(2,1)

numEpochs = 5000
learningRate = 0.0002

for epoch in range(numEpochs):
    xbShuffled, yShuffled = shuffle(xb, y) # shuffle your dataset at the beginning of each epoch.
    for i in range(0, xbShuffled.shape[0], batchsize):
        xbi = xbShuffled[i:i+batchsize]
        yi = yShuffled[i:i+batchsize]
        gradient = 2/batchsize*xbi.T.dot(xbi.dot(theta)-yi)
        theta = theta - learningRate*gradient
        
print(theta)
# read data
advancedData = pd.read_csv('../input/advanced.csv')
advancedData.head()
plt.plot('fixed acidity','pH', data = advancedData, marker = 'o', linestyle = '') # fixed acidity, pH
plt.xlabel('fixed acidity', fontsize = 18)
plt.ylabel('pH', fontsize = 18)
plt.show()
reg = LinearRegression()
x = advancedData['fixed acidity'].as_matrix().reshape(-1,1)
y = advancedData['pH'].as_matrix().reshape(-1,1)
reg.fit(x,y)
xFit = np.linspace(4,16,num=100).reshape(-1,1)
yFit = reg.predict(xFit)
plt.plot('fixed acidity','pH', data = advancedData, marker = 'o', linestyle = '') 
plt.plot(xFit,yFit, color = 'r',lw=3)
plt.xlabel('fixed acidity', fontsize = 18)
plt.ylabel('pH', fontsize = 18)
plt.show()
# compute MSE
xb = np.c_[np.ones((len(x),1)),x]
theta = np.array([reg.intercept_[0],reg.coef_[0][0]]).reshape(-1,1)

linMSE = MSE(xb,y,theta)
print(linMSE)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias = False)
xPoly = poly.fit_transform(x)
print(xPoly[0:5])
reg = LinearRegression()
reg.fit(xPoly,y)
print(reg.intercept_, reg.coef_)
xFit=np.linspace(4,16,num=100).reshape(-1,1)
xFit=poly.fit_transform(xFit)
#print(xFit[0:5])
yFit = reg.predict(xFit)
plt.plot('fixed acidity','pH', data = advancedData, marker = 'o', linestyle = '') 
plt.plot(xFit[:,0],yFit, color = 'r',lw=3)
plt.xlabel('fixed acidity', fontsize = 18)
plt.ylabel('pH', fontsize = 18)
plt.show()
# compute MSE
xb = np.c_[np.ones((len(x),1)),xPoly]
theta = np.array([reg.intercept_[0],reg.coef_[0][0],reg.coef_[0][1]]).reshape(-1,1)

polyMSE = MSE(xb,y,theta)
print(polyMSE)
poly = PolynomialFeatures(degree=20, include_bias = False)
xPoly = poly.fit_transform(x)
#print(xPoly[0:1])

reg = LinearRegression()
reg.fit(xPoly,y)

xFit = np.linspace(4,16,num=100).reshape(-1,1)
xFit = poly.fit_transform(xFit)
yFit = reg.predict(xFit)
plt.plot('fixed acidity','pH', data = advancedData, marker = 'o', linestyle = '') 
plt.plot(xFit[:,0],yFit, color = 'r',lw=3)
plt.xlabel('fixed acidity', fontsize = 18)
plt.ylabel('pH', fontsize = 18)
plt.show()