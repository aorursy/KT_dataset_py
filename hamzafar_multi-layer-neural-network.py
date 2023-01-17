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
def initialize_paramaters(layer_dims, pt = False):
    # return weights and bias of required network shape
        #layer_dims: layer dim
        # pt: wehter to print shapes
    parameters = {}
    L = len(layer_dims)
    for l in range(1,L):
        parameters['w'+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * 0.01
        parameters['b'+str(l)] = np.zeros(shape=(layer_dims[l], 1))
        
        assert(parameters['w' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
        if(pt == True):
            print('w'+str(l))
            print(layer_dims[l],layer_dims[l-1])
    return(parameters)
def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s
def relu(z):
    # Takes input as z and return relu of value    
    r = np.maximum(0, z)
    return r
def single_layer_forward_pass(prev_a, w, b, act_fun):
    # return activations a and z
        # prev_a: last layer activations values
        # w, b : parameters weight and bias
        # act_func: either sigmoid or relu
    z = np.dot(w,prev_a) + b
    a = np.zeros((z.shape))
    if(act_fun == 'sigmoid'):
        a = sigmoid(z)
    elif(act_fun == 'relu'):
        a = relu(z)
    return(a,z)
def multi_layer_forward_pass(x, parameters, layer_dims, act_fun, pt = False):
    # return activations at output layer
        # x: input data
        # parameters: dictonary object of weights and bias
        # layer_dims: layer dim
        # act_fun: activation function either relu or sigmoid
        # pt: wehter to print shapes
    L = len(layer_dims)
    prev_a = x
    activations = {}
    for l in range(1,L):
        if(l == L-1):
#             print('last layer')
            w = parameters['w'+str(l)]
            b = parameters['b'+str(l)]
            a, z = single_layer_forward_pass(prev_a, w, b, 'sigmoid')
            activations['a'+str(l)] = a
            
            assert((a.shape) == (w.shape[0], prev_a.shape[1]))
            
            if(pt == True):
                print('w'+str(l))
                print(a.shape)
        else:
            w = parameters['w'+str(l)]
            b = parameters['b'+str(l)]
            a, z = single_layer_forward_pass(prev_a, w, b, act_fun)
            prev_a = a
            if(pt == True):
                print('w'+str(l))
                print(a.shape)
            activations['a'+str(l)] = a
            
            assert((a.shape) == (w.shape[0], prev_a.shape[1]))
            
    return activations
def compute_cost(y, activations, layer_dims, m):
    # calculate cost w.r.t. activation activations and actual value y
        # y: target value
        # activations: dictionary of activations
        # layer_dims: network layers dimensions
        # m: number of training examples in data set
    L = len(layer_dims)   
    a = activations['a'+ str(L-1)]
    loss = -1*(y* np.log(a) + (1-y) * np.log(1-a))
    cost = np.sum(loss)/m
    return(cost)
def sigmoid_deri(a):
    # return partial derivative of sigmoid
        #a: activation
    da = (1-a) * a
    return da
def relu_deri(a):
    # return partial derivative of relu
        #a: activation
    da = np.array(a, copy=True)
    da[a <= 0] = 0
    return da
def derivaive_parameters(dz, a):
    # compute partial derivatives of w and b
        # dz: partial derivative of z
        # a: activations
    dw = (1 / m) * np.dot(dz, a.T)
    db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
    return(dw, db)
def derivaive_z(dz, w, da):
    # computes derivative of z
        # dz: next node dz
        # w: paramter weight of next node
        # da: partial derivative of activation functions.
    dz = np.dot(w.T, dz) * da
    return dz
def back_propagate(x, y, m, parameters, activations, layer_dims, act_fun):
    # return gradeint of paramters
        # x: input data
        # y: target label
        # m: number of training examples in data set
        # parameters: dictonary object of weights and bias
        # activations: dictionary object containing activation on different layers
        # layer_dims: layer dim
        # act_fun: activation function either relu or sigmoid

    L = len(layer_dims)
    grads = {}
    grads_line = {}

    for l in reversed(range(L-1)):
        l = l+1
        if(l == L-1):
#             print('a'+str(l))
    #         print(activations['a'+str(l+1)])
            grads_line['dz'+str(l)] = activations['a'+str(l)] - y
        
            grads['dw'+str(l)], grads['db'+str(l)] = derivaive_parameters(grads_line['dz'+str(l)], activations['a'+str(l-1)])
            
            assert(grads['dw'+str(l)].shape) == (parameters['w'+str(l)].shape)
            assert(grads['db'+str(l)].shape) == (parameters['b'+str(l)].shape)
        else:
#             print('a'+str(l))
            da = np.empty(activations['a'+str(l)].shape)

            if(act_fun == 'sigmoid'):
                da = sigmoid_deri(activations['a'+str(l)])            
            elif(act_fun == 'relu'):
                da = relu_deri(activations['a'+str(l)])

            grads_line['dz'+str(l)] = derivaive_z(grads_line['dz'+str(l+1)], parameters['w'+str(l+1)], da)
        
            if(l-1 != 0):
                grads['dw'+str(l)], grads['db'+str(l)] = derivaive_parameters(grads_line['dz'+str(l)], activations['a'+str(l-1)])
            else:
                 grads['dw'+str(l)], grads['db'+str(l)]  = derivaive_parameters(grads_line['dz'+str(l)], x)
                    
            assert(grads['dw'+str(l)].shape) == (parameters['w'+str(l)].shape)
            assert(grads['db'+str(l)].shape) == (parameters['b'+str(l)].shape)
                
    return grads
def update_parameters(parameters, grads, layer_dims):
    # update and return parameters 
        # parameters: dictonary object of w's and b's
        # grads: dictionary: object of gradient of w's and b's
        # layer_dims: layer dim

    L = len(layer_dims)
    for l in range(1,L):
#         print(l)
        parameters['w'+str(l)] = parameters['w'+str(l)] -(lr * grads['dw'+str(l)])
        parameters['b'+str(l)] = parameters['b'+str(l)] -(lr * grads['db'+str(l)])
    return parameters    
def optimize_parameters(x, y, parameters, act_fun, layer_dims, m, num_iter):
    lst_cost = []
    
    for i in range(num_iter):
        activations = multi_layer_forward_pass(x, parameters, layer_dims, act_fun, pt = False)
        cost = (compute_cost(y, activations, layer_dims, m))
        grads = back_propagate(x, y, m, parameters, activations, layer_dims, act_fun)
        parameters = update_parameters(parameters, grads, layer_dims)
        lst_cost.append(cost)
#         lst_cost.append(format(cost, '.4f'))
    
    return (lst_cost, parameters)
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
n_x = 50
n_y = 1
m = 10000
lr = 0.5
num_iter = 30

x = generate_bits(n_x, m)
y = generate_label(x, m)

layer_dims = [n_x,4,3,n_y]
act_fun = 'relu'

parameters = initialize_paramaters(layer_dims, pt = False)

lst_cost_s, parametes_s = optimize_parameters(x, y, parameters, act_fun, layer_dims, m, num_iter)

# plt_res(lst_cost, 'cost', lr)

m = 100000
x = generate_bits(n_x, m)
y = generate_label(x, m)

parameters = initialize_paramaters(layer_dims, pt = False)
lst_cost_m, parametes_m = optimize_parameters(x, y, parameters, act_fun, layer_dims, m, num_iter)


m = 1000000
x = generate_bits(n_x, m)
y = generate_label(x, m)

parameters = initialize_paramaters(layer_dims, pt = False)
lst_cost_l, parametes_l = optimize_parameters(x, y, parameters, act_fun, layer_dims, m, num_iter)

def get_prediction(x, parameters, layer_dims, act_fun,  m):
    # returns the prediction on the dataset
        # x: input data (unseen)
        # w, b: parameters weights and bias
        # m: total sample set
    L = len(layer_dims)   
    
    activations = multi_layer_forward_pass(x, parameters, layer_dims, act_fun, pt = False)
    a = activations['a'+ str(L-1)]
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

y_prediction = get_prediction(x, parametes_s, layer_dims, act_fun,  tm)
acc_s = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, parametes_m, layer_dims, act_fun,  tm)
acc_m = get_accuracy(y, y_prediction, tm)

y_prediction = get_prediction(x, parametes_l, layer_dims, act_fun,  tm)
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