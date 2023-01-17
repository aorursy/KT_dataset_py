%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
matplotlib.rc('xtick', labelsize=14) 
matplotlib.rc('ytick', labelsize=14)
def generate_data(n):
    d = 100
    w = np.zeros(d)
    for i in range(0,10):
        w[i] = 1.0
    #
    trainx = np.random.normal(size=(n,d))
    e = np.random.normal(size=(n))
    trainy = np.dot(trainx, w) + e
    #
    return trainx, trainy
def ridge_regression_GD(x,y,C):
    y=np.reshape(y,(y.shape[0],1))
    d = x.shape[1] + 1
    xtemp = np.zeros((x.shape[0],d))
    xtemp[:,1:] = x
    xtemp[:,0] = 1
    w = np.zeros((d,1))
    for i in range(11):
        w[i,0]= 1
    ##
    
    losses = []
    for i in range(1,100):
        #step = step*3
        wt = w
        #print("step is " + str(i))
        #print(w.shape)
        temp1 = np.dot(xtemp,w)
        temp2 = np.subtract(y,temp1)
        temp3 = np.multiply(temp2,xtemp)
        #print("shape of temp3 will be " )
        #print(temp3.shape)
        k  = np.sum(temp3,axis=0)
        k = np.reshape(k,(k.shape[0],1))
        #print("shape of k")
        #print(k.shape)
        w = w + 2*0.003*k-2*0.03*C*w
        
        l= np.sum(y - np.dot(xtemp,w))
        losses.append(l)
    b = w[0,0]
    losses = np.array(losses)
    #print(losses.shape)
    w = w[1:,:]
    
    ##
    return w,b,losses
# Generate 200 data points
n = 200
x,y = generate_data(n)
# Set regularization constant

    # Run gradient descent solver
#for i in range(10):
C = 10
w, b, losses = ridge_regression_GD(x,y,C)
# Plot the losses
plt.plot(losses,'r')
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.show()
w.shape
def compute_mse(w,b,x,y):
    residuals = y - (np.dot(x, w) + b)
    return np.dot(residuals, residuals)/n
# Generate 200 data points
n = 200
x,y = generate_data(n)
# Set regularization constant
C = 10.0
# Run gradient descent solver and compute its MSE
w, b, losses = ridge_regression_GD(x,y,C)
print(w.shape)
print(b.shape)
print(losses.shape)
# Use built-in routine for ridge regression and compute MSE
regr = linear_model.Ridge(alpha=C)
regr.fit(x, y)
# Print MSE values and L2 distance between the regression functions
print ("MSE of gradient descent solver: ", compute_mse(w,b,x,y))
print ("MSE of built-in solver: ", mean_squared_error(regr.predict(x), y))
print ("Distance between w-coefficients: ", np.linalg.norm(w-regr.coef_))
