# This Python 3 environment comes with many helpful analytics libraries installed
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting graph

def generate_bits(n_x, m):
# Generate a m x n_x array of ints between 0 and 1, inclusive:
# m: number of rows
# n_x : number of columns per rows/ feature set
    np.random.seed(1)
    data = np.random.randint(2, size=(n_x, m))
    return(data)
def generate_label(data, m):
    # generate label by appyling xor operation to individual row
    # return list of label (results)
        # data: binary data set of m by n_x size
    lst_y = []
    y= np.empty((m,1))
    k = 0
    for tmp in data.T:
        xor = np.logical_xor(tmp[0], tmp[1])

        for i in range(2, tmp.shape[0]):
            xor = np.logical_xor(xor, tmp[i])
    #     print(xor)
        lst_y.append(int(xor))
        y[k,:] = int(xor)
        k+=1
    return(y.T)
def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s
def intialize_param(n_x):
    # initialize paramaters w and b to zero and return them
    # size of w equal to size fo feature set and b is single value
        # n_x: size input feature    
    w = np.zeros(shape=(n_x, 1))
    b = 0
    return(w,b)
def get_activation_loss(x, w, b):
    # this function return action, cost and z values
        # x: input data
        # w: weights
        # b: bias
    z = np.dot(w.T, x) + b
    a = sigmoid(z)

    cost = (1/m) * np.sum(-1 * (y * np.log(a) + (1 - y) * (np.log(1 - a))))
    return(a,cost, z)
def update_paramters(x, w, b, a, y, lr, m):
    # find the gradient of paramaters and update them (w and b)
        # x: input data 
        # w, b: parameters (w and b)
        # a, y: activation and actual values
        # m, lr: total number of rows, learning rate
    dw = (1/m) * np.dot(x,(a-y).T)
    db = (1/m) * np.sum(a - y)
    
    w = w - (lr*dw)
    b = b - (lr*db)
    
    return(w, b)
def plt_res(lst, ylab, lr):
    #This will plot the list of values at y axis while x axis will contain number of iteration
    #lst: lst of action/cost
    #ylab: y-axis label
    #lr: learning rate
    plt.plot(lst)
    plt.ylabel(ylab)
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()
def optimize_paramters(x, y, w, b, n_x, lr, num_iter):
    # this function returns upadated values of parameters and cost
    # It first initialize parameters and update them by computing partial derivatives
    # Then loop over 
        # x: input data
        # y: actual values (labels)
        # w, b: parameters
        # n_x, lr: input feature length, learning rate
        # num_iter: number of cycle
    lst_cost = []

    w, b = intialize_param(n_x)

    for i in range(num_iter):
        a,cost,z = get_activation_loss(x, w, b)
#         print('cost after iteration %i: %f' %(i,cost))
        w, b = update_paramters(x, w, b, a, y, lr, m)
        lst_cost.append(cost)
    
    return(w, b, lst_cost)
n_x = 50
m = 10000
num_iter = 1000
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
lr = 0.07

#w_s, b_s, lst_cost_s represent values when sample set is 10000
w_s,b_s, lst_cost_s = optimize_paramters(x, y, w, b, n_x, lr, num_iter)
##----------##

m = 100000
# num_iter = 150
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
# lr = 0.07

#w_m, b_m, lst_cost_m represent values when sample set is 100000
w_m,b_m, lst_cost_m = optimize_paramters(x, y, w, b, n_x, lr, num_iter)
##----------##

m = 1000000
num_iter = 15
w, b = intialize_param(n_x)
x = generate_bits(n_x,m)
y = generate_label(x, m)
# lr = 0.07

#w_l, b_l, lst_cost_l represent values when sample set is 1000000
w_l,b_l, lst_cost_l = optimize_paramters(x, y, w, b, n_x, lr, num_iter)
def get_prediction(x, w, b, m):
    # returns the prediction on the dataset
        # x: input data (unseen)
        # w, b: parameters weights and bias
        # m: total sample set
    a = sigmoid(np.dot(w.T, x) + b)
    y_prediction = np.zeros((1, m))
    for i in range(a.shape[1]):
        y_prediction[0,i] = 1 if a[0, i] > 0.5 else 0
    return(y_prediction)
def get_accuracy(y, y_prediction, m):
    # return the accuracy by calculated the difference between actual and predicted label
        # y: actual values
        # y_prediction: prediction acquired from the get_prediction
        # m: total number of sample
    df = pd.DataFrame()
    df['actual'] = y[0]
    df['prediction'] = y_prediction[0]
    df['compare']= df['prediction'] == df['actual']

#     print(df[df['compare']==True])
#     print('Accuracy: ' ,len(df[df['compare']==True]['compare'])/m)
    return(len(df[df['compare']==True]['compare'])/m)
tm = int(0.1 * m)
x = generate_bits(n_x, tm)
y = generate_label(x, tm)

y_prediction = get_prediction(x, w_s, b_s, tm)
acc_s = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, w_m, b_m, tm)
acc_m = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, w_l, b_l, tm)
acc_l = get_accuracy(y, y_prediction, tm)
print('------- 10000 training set-------------')
print('Accurcy at 10000 training set: ', acc_s)
plt_res(lst_cost_s, 'cost', lr)

print('-------100000 training set-------------')
print('Accurcy at 100000 training set: ', acc_m)
plt_res(lst_cost_m, 'cost', lr)

print('-------1000000 training set-------------')
print('Accurcy at 1000000 training set: ', acc_l)
plt_res(lst_cost_l, 'cost', lr)