
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import linear_model
iris = datasets.load_iris()
iris.feature_names
iris.target_names
X = iris.data[:, :2]

y_bool = iris.target!=2

y = iris.target[y_bool]

X = X[y_bool]
y.shape, X.shape
iris.feature_names
iris.target_names
plt.figure(1,figsize=(8,6))
ax = plt.scatter(X[:,0],X[:,1],c=y, cmap="bwr_r")
plt.xlabel(iris.feature_names[0], fontsize=15)
plt.ylabel(iris.feature_names[1], fontsize=15)
s1 = plt.scatter([],[], s=30, marker='o', c='red' )
s2 = plt.scatter([],[], s=30, marker='o', c='blue')

plt.legend((s1,s2),
       (iris.target_names[0],iris.target_names[1]),
       scatterpoints=1,
       loc='upper right',
       fontsize=12,
           )
plt.show()
X = np.c_[np.ones((X.shape[0],1)), X[:]]
y = y.reshape(-1,1)


# Parameters required for Gradient Descent
alpha = 0.1   #learning rate
m = y.size  #no. of samples
np.random.seed(10)
theta = np.random.rand(3)  #initializing theta with some random values
theta = theta.reshape(-1,1)
def gradient_descent(x, y, m, theta,  alpha):
    cost_list = []   #to record all cost values to this list
    theta_list = []  #to record all theta_0 and theta_1 values to this list 
    prediction_list = []
    run = True
    cost_list.append(1e10)    #we append some large value (initial value) to the cost list
    i=0
    while run:
        Z = np.dot(x, theta) 
        prediction = 1 / (1 + np.exp(-Z))   #predicted y values - sigmoid function
        prediction_list.append(prediction)
        error = prediction - y
        cost = np.sum(-(y * np.log(prediction) + (1 - y) * np.log(1 - prediction))) / m   #  (1/2m)*sum[(error)^2]
#         cost = -(1/m)*np.sum(np.dot(y.T, np.log(prediction)) + np.dot((1 - y).T, np.log(1 - prediction)))
        
        cost_list.append(cost)
        theta = theta - (alpha * (1/m) * np.dot(x.T, error))   # alpha * (1/m) * sum[error*x]
        theta_list.append(theta)
        if cost_list[i]-cost_list[i+1] < 1e-8:   #checking if the change in cost function is less than 10^(-8)
            run = False

        i+=1
    cost_list.pop(0)   # Remove the large number we added in the begining 
    return prediction_list, cost_list, theta_list
%%time 
prediction_list, cost_list, theta_list = gradient_descent(X, y, m, theta, alpha)
theta = theta_list[-1]
theta
plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.plot(cost_list)
plt.show()
xx = np.linspace(X.min(), X.max())
yy = - theta[1]/theta[2] *xx - theta[0]/theta[2]
X.shape
X_setosa = X[:,1:][y.ravel()==0] 
X_versicolor = X[:,1:][y.ravel()==1]
plt.figure(1,figsize=(8,6))
plt.plot(X_setosa[:,0], X_setosa[:,1],'o',label='setosa')
plt.plot(X_versicolor[:,0], X_versicolor[:,1],'o',label='versicolor')
plt.plot(xx,yy,label='Hyperplane-scratch')
plt.xlim(X[:,1].min()-0.1, X[:,1].max()+0.1)
plt.ylim(X[:,2].min()-0.1, X[:,2].max()+0.1)
plt.legend()
plt.xlabel(iris.feature_names[0], fontsize=15)
plt.ylabel(iris.feature_names[1], fontsize=15)
plt.show()
theta
X_sklearn = iris.data[:, :2]

y_bool = iris.target!=2

y = iris.target[y_bool]

X_sklearn = X_sklearn[y_bool]
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e100, penalty='l1', tol=1e-8, solver='liblinear')
lr = lr.fit(X_sklearn,y)
lr.coef_
lr.intercept_
yy_l1 = - lr.coef_[0,0]/lr.coef_[0,1] *xx - lr.intercept_[0]/lr.coef_[0,1]
plt.figure(1,figsize=(8,6))
plt.plot(X_setosa[:,0], X_setosa[:,1],'o',label='setosa')
plt.plot(X_versicolor[:,0], X_versicolor[:,1],'o',label='versicolor')
plt.plot(xx,yy,label='Hyperplane-scratch')
plt.plot(xx,yy_l1,'--',label='Sklearn')
plt.xlim(X[:,1].min()-0.1, X[:,1].max()+0.1)
plt.ylim(X[:,2].min()-0.1, X[:,2].max()+0.1)
plt.legend()
plt.xlabel(iris.feature_names[0], fontsize=15)
plt.ylabel(iris.feature_names[1], fontsize=15)
plt.show()
'Theta_0 and Theta_1 are {},{}'.format(round(theta[0,0],3),round(theta[1,0],3))
'Theta_0 and Theta_1 are {},{}'.format(round(lr.intercept_[0],3), round(lr.coef_[0,0],3))
test_data = [[5,3.5]]
iris.target_names[lr.predict(test_data)[0] ]