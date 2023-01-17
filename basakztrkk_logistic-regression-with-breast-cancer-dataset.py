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
import matplotlib.pyplot as plt
# data implement
data = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
# data.head()
# data cleaning
data.drop(["Unnamed: 32","id"], axis=1, inplace=True)
data.diagnosis = [ 1 if each == "M" else 0 for each in data.diagnosis]
# print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis = 1)
# normalization
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

# %% Train test split

from sklearn.model_selection import train_test_split

# %25 test, %75 train
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
#%% initialize parameter 

def initialize_weight_bias(dimension):
    
    w = np.full((dimension,1),0.01)  # weight
    b = 0.0                          # bias 
    return w,b

#%% sigmoid function 

def sigmoid(z):
    
    y_head = 1/(1 + np.exp(-z))
    return y_head
# %%
    
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


#%% Updating(learning) parameters
    
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: Cost %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
    

#%%  # prediction
    
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression
    
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weight_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 400)   
#%%   LR with Sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix



lr = LogisticRegression(random_state = 42, max_iter = 400,)
lr.fit(x_train.T,y_train.T)


p_pred = lr.predict_proba(x_test.T)
y_pred = lr.predict(x_test.T)
score_ = lr.score(x_test.T,y_test.T)
conf_m = confusion_matrix(y_test.T, y_pred).T
report = classification_report(y_test, y_pred)

print("Test proba:\n", p_pred )


print("Test accuracy {}".format(score_))


print("Confusion matrix:\n",conf_m )


print("Test report:\n", report )
