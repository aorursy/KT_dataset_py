# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
data = pd.read_excel('../input/acute-liver-failure/ALF_Data.xlsx')
data["Source of Care"]=data["Source of Care"].apply(lambda x: np.NaN if str(x).isspace() else x)
data.dropna(axis=0, how='any', inplace=True)

data.info()
#加入虚拟变量#############################ADD Dummy Variables
dummy_ranks1 = pd.get_dummies(data['Gender'], prefix='gender')
dummy_ranks2 = pd.get_dummies(data['Region'], prefix='Region')
dummy_ranks3 = pd.get_dummies(data['Source of Care'], prefix='Source')
data=data.join(dummy_ranks1["gender_F"])
data=data.join(dummy_ranks2.loc[:, 'Region_north':])
data=data.join(dummy_ranks3.loc[:, 'Source_Never Counsulted':])
data=data.drop(['Gender','Region','Source of Care'], axis=1)

data.loc[data["ALF"]==0]
data1=data.loc[data["ALF"]==0].sample(n=len(data.loc[data["ALF"]==1])).append(data.loc[data["ALF"]==1])
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s
#初始化参数值
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b
#前向传导
def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # compute cost
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}
    return grads, cost
#后向传导
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 2000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs
#预测    
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if (A[0][i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
#整合函数
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    Returns:
    d -- dictionary containing information about the model.
    """
    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d
###############################

x=data1.drop(['ALF'], axis=1)
y=data1["ALF"]
#划分训练集测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=2)
m_train = x_train.shape[0]
m_test = x_test.shape[0]
train_x=x_train.T
test_x=x_test.T
train_y=np.squeeze(np.asarray(y_train))
test_y=np.squeeze(np.asarray(y_test))
d = model(train_x,train_y,test_x,test_y, num_iterations = 30000, learning_rate = 0.00005, print_cost = True)
#计算 fptptnfn
TP=0
FP=0
TN=0
FN=0

for i in range(len(test_y)):
    if d['Y_prediction_test'][0][i]==1 and test_y[i]==1:
        TP+=1
    elif d['Y_prediction_test'][0][i]==1 and test_y[i]==0:
        FP+=1
    elif d['Y_prediction_test'][0][i]==0 and test_y[i]==0:
        TN+=1
    elif d['Y_prediction_test'][0][i]==0 and test_y[i]==1:
        FN+=1
P=TP/(FP+TP) 
R=TP/(TP+FN) 
print("Precision="+str(P)+"\n"+"Recall="+str(R))
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
x_test_all=data.drop(['ALF'], axis=1)
y_test_all=data["ALF"]
m_test = x_test_all.shape[0]
test_all_x=x_test_all.T
test_all_y=np.squeeze(np.asarray(y_test_all))
d = model(train_x,train_y,test_all_x,test_all_y, num_iterations = 20000, learning_rate = 0.00005, print_cost = True)
#计算 fptptnfn
TP=0
FP=0
TN=0
FN=0
for i in range(len(test_all_y)):
    if d['Y_prediction_test'][0][i]==1 and test_all_y[i]==1:
        TP+=1
    elif d['Y_prediction_test'][0][i]==1 and test_all_y[i]==0:
        FP+=1
    elif d['Y_prediction_test'][0][i]==0 and test_all_y[i]==0:
        TN+=1
    elif d['Y_prediction_test'][0][i]==0 and test_all_y[i]==1:
        FN+=1
P=TP/(FP+TP)#20.87%
R=TP/(TP+FN)#84.67%
print("Precision="+str(P)+"\n"+"Recall="+str(R))
x=data1[["Age","Height","Good Cholesterol","Bad Cholesterol","Physical Activity","HyperTension","Diabetes","gender_F"]]
y=data1["ALF"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20200520)
m_train = x_train.shape[0]
m_test = x_test.shape[0]
train_x=x_train.T
test_x=x_test.T
train_y=np.squeeze(np.asarray(y_train))
test_y=np.squeeze(np.asarray(y_test))
d = model(train_x,train_y,test_x,test_y, num_iterations = 20000, learning_rate = 0.00005, print_cost = True)
#计算 fptptnfn
TP=0
FP=0
TN=0
FN=0
for i in range(len(test_y)):
    if d['Y_prediction_test'][0][i]==1 and test_y[i]==1:
        TP+=1
    elif d['Y_prediction_test'][0][i]==1 and test_y[i]==0:
        FP+=1
    elif d['Y_prediction_test'][0][i]==0 and test_y[i]==0:
        TN+=1
    elif d['Y_prediction_test'][0][i]==0 and test_y[i]==1:
        FN+=1
print(TP,FP,TN,FN)
#计算P R
P=TP/(FP+TP)
R=TP/(TP+FN)
print("Precision="+str(P)+"\n"+"Recall="+str(R))