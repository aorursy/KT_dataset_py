
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model
iris = datasets.load_iris()
iris.feature_names
iris.target_names
X = iris.data[:, 0]

y_bool = iris.target!=2

y = iris.target[y_bool]

X = X[y_bool]
plt.scatter(X, y)
plt.xlabel('Sepal Length ', fontsize=15)

plt.ylabel('0 - setosa, 1 - versicolor ', fontsize=15)
plt.show()
X = np.c_[np.ones((X.shape[0],1)), X[:]]
y = y.reshape(-1,1)


# Parameters required for Gradient Descent
alpha = 0.1   #learning rate
m = y.size  #no. of samples
np.random.seed(10)
theta = np.random.rand(2)  #initializing theta with some random values
theta = theta.reshape(-1,1)
def gradient_descent(x, y, m, theta,  alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list 
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value to the cost list
    i=0
    while run:
        Z = np.dot(x, theta) 
        prediction = 1 / (1 + np.exp(-Z))   #predicted y values 
        prediction_list.append(prediction)
        error = prediction - y
        cost = np.sum(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))) / m   #  (1/2m)*sum[(error)^2]
        
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-9:   #checking if the change in cost function is less than 10^(-9)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    return prediction_list, cost_list, theta_list
prediction_list, cost_list, theta_list = gradient_descent(X, y, m, theta, alpha)
theta = theta_list[-1]
plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()
theta
X_test = np.linspace(4, 7, 300)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
loss = sigmoid(X_test*theta[1] + theta[0])
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.plot(X_test, loss, c='C0', label='Hyperplane')
plt.scatter(X[:,1], y, c='C1', label='Training data')
# plt.plot(X_test,X_test*theta[1] + theta[0] )
# plt.axhline(0.5, c='C2',label='0.5 Threshold')
# plt.axvline(5.4147157190635449, c='C2',label='0.5 Threshold')
# plt.axvline
plt.legend()
plt.xlabel('Sepal Length (Input)', fontsize=15)
plt.ylabel("Probability of the output" "\n" "(0 - setosa, 1 - versicolor)", fontsize=15)
plt.show()
# 0.5 threshold corresponds to 
boundary = X_test[np.where(loss >= 0.5)[0][0]]
print(round(boundary,3))
plt.figure(1, figsize=(8, 6))
plt.clf()
plt.plot(X_test, loss, c='C0', label='Hyperplane')
plt.scatter(X[:,1], y, c='C1', label='Training data')
plt.axhline(0.5, c='C2',label='0.5 Threshold', linewidth=0.7)
plt.axvline(boundary, c='C3',label='Boundary', linewidth=0.7)
plt.legend()
plt.xlabel('Sepal Length (Input)', fontsize=15)
plt.ylabel("Probability of the output" "\n" "(0 - setosa, 1 - versicolor)", fontsize=15)
plt.show()
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e100)
X.shape
X[:5]
lr = lr.fit(X[:,1].reshape(-1,1),y.ravel())
'Theta_0 and Theta_1 are {},{}'.format(round(lr.intercept_[0],3), round(lr.coef_[0,0],3))
theta[0,0], theta[1,0]
'Theta_0 and Theta_1 are {},{}'.format(round(theta[0,0],3),round(theta[1,0],3))