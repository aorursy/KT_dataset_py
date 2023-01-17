# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')
df1 = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

#df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
df2 = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')
def conv_forward(A_prev, W, b, hyperparam, stride):
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    (f, f, n_C_prev, n_C) = W.shape
    
    #stride = hyperparam['stride']
    pad = hyperparam['pad']
    
    n_H = int(((n_H_prev + 2 * pad - f) / stride) + 1)
    n_W = int(((n_W_prev + 2 * pad - f) / stride) + 1)
    
    Z = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        
        for h in range(n_H):
            v_start = h * stride
            v_end = h * stride + f
            
            for w in range(n_W):
                h_start = w * stride
                h_end = w * stride + f
                
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[v_start:v_end, h_start:h_end,:]
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i,h,w,c] = np.sum(a_slice_prev * weights) + float(biases)
                    
    assert(Z.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, W, b, hyperparam, stride)
    
    return Z, cache
            
def pool_forward(A_prev, hyperparam, stride, mode='max'):
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hyperparam['f']
    #stride = hyperparam['stride']
    n_H = int((n_H_prev - f) / stride + 1)
    n_W = int((n_W_prev - f) / stride + 1)
    n_C = n_C_prev
    A = np.zeros((m, n_H, n_W, n_C))
    
    for i in range(m):
        for h in range(n_H):
            v_start = h * stride
            v_end = h * stride + f
        
            for w in range(n_W):
                h_start = w * stride
                h_end = w * stride + f
                
                for c in range(n_C):
                    a_prev_slice = A_prev[i, v_start:v_end, h_start:h_end, c]
                    
                    if mode == 'max':
                        A[i, h, w, c] = np.amax(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.average(a_prev_slice)
                        
    
    assert(A.shape == (m, n_H, n_W, n_C))
    
    cache = (A_prev, hyperparam, stride)
    
    return A, cache
    
def activation_forward(Z, mode='relu'):
    cache = Z
    if mode == 'relu':
        return np.maximum(Z, 0), cache
    elif mode == 'tanh':
        return np.tanh(Z), cache
    
    
def FC_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
def conv_backward(dZ, cache):
    
    (A_prev, W, b, hyperparam, stride) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    #stride = hyperparam['stride']
    pad = hyperparam['pad']
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))
    dA_prev_pad = np.pad(dA_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))    
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f

                    a_slice = a_prev_pad[v_start:v_end, h_start:h_end, :]
                    da_prev_pad[v_start:v_end, h_start:h_end, :] += W[:,:,:,c] * float(dZ[i,h,w,c])
                    dW[:,:,:,c] += a_slice * float(dZ[i,h,w,c])
                    db[:,:,:,c] += float(dZ[i,h,w,c])
        if pad == 0:
                dA_prev[i,:,:,:] = da_prev_pad
        else:
            dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad,:]
        
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db
    
def pool_backward(dA, cache, mode='max'):
    
    (A_prev, hyperparam, stride) = cache
    #stride = hyperparam['stride']
    f = hyperparam['f']
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    
    for i in range(m):
        a_prev = A_prev[i]
        
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v_start = h * stride
                    v_end = h * stride + f
                    h_start = w * stride
                    h_end = w * stride + f

                    if mode == 'max':
                        a_prev_slice = a_prev[v_start:v_end, h_start:h_end, c]
                        
                        mask = (a_prev_slice == np.max(a_prev_slice))

                        dA_prev[i, v_start:v_end, h_start:h_end, c] += mask * dA[i, h, w, c]
                        
                    elif mode == 'average':

                        dA_prev[i, v_start:v_end, h_start:h_end, c] += (dA[i,h,w,c] / (f * f)) * np.ones((f, f))

    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dTanh(x):
    return 1.0 - np.tanh(x)**2

def activation_backward(dA, cache, mode='relu'):
    Z = cache
    if mode == 'relu':
        dZ = dA * dRelu(Z)
    elif mode == 'tanh':
        dZ = dA * dTanh(Z)
    return dZ

def FC_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db
def nloss(Z, y):
    
    n = Z.shape[1]
    
    softmax = np.exp(Z) / (np.sum(np.exp(Z), axis=0, keepdims=True))
    
    if n == 2:
        return - (1 / n) * np.sum(y * np.log(softmax) + (1 - y) * np.log(1 - softmax), keepdims=True), softmax
    else:
        return - (1 / n) * np.sum((y * np.log(softmax))), softmax
def initialize_parameters():
    C_W1 = np.random.randn(5, 5, 1, 8) * (1 / np.sqrt(28 * 28))
    C_b1 = np.zeros((1, 1, 1, 8))
    
    C_W2 = np.random.randn(5, 5, 8, 16) * (1 / np.sqrt(12 * 12 * 16))
    C_b2 = np.zeros((1, 1, 1, 16))
    
    FC_W1 = np.random.randn(120, 256) * (1 / np.sqrt(256))
    FC_b1 = np.zeros((120, 1))
    
    FC_W2 = np.random.randn(10, 120) * (1 / np.sqrt(120))
    FC_b2 = np.zeros((10, 1))
    
    hyperparam = {
        'stride': 1,
        'pad': 0,
        'f': 2
    }
    
    parameters = {
        'C_W1': C_W1,
        'C_b1': C_b1,
        'C_W2': C_W2,
        'C_b2': C_b2,
        'FC_W1': FC_W1,
        'FC_b1': FC_b1,
        'FC_W2': FC_W2,
        'FC_b2': FC_b2
    }
    return parameters, hyperparam
def forward_step(train_set_X, Y, params, hyperparam):
    model_cache ={}
    Z, cache_0 = conv_forward(train_set_X, params['C_W1'], params['C_b1'], hyperparam, 1)
    model_cache['0'] = cache_0

    A, cache_1 = activation_forward(Z, mode='tanh')
    model_cache['1'] = cache_1

    A, cache_2 = pool_forward(A, hyperparam, 2, mode='max')
    model_cache['2'] = cache_2

    Z, cache_3 = conv_forward(A, params['C_W2'], params['C_b2'], hyperparam, 1)
    model_cache['3'] = cache_3

    A, cache_4 = activation_forward(Z, mode='tanh')
    model_cache['4'] = cache_4

    A, cache_5 = pool_forward(A, hyperparam, 2, mode='max')
    model_cache['5'] = cache_5
    
    A = A.reshape(-1, 256).T
    
    Z, cache_6 = FC_forward(A, params['FC_W1'], params['FC_b1'])
    model_cache['6'] = cache_6

    A, cache_7 = activation_forward(Z, mode='tanh')
    model_cache['7'] = cache_7

    Z, cache_8 = FC_forward(A, params['FC_W2'], params['FC_b2'])
    model_cache['8'] = cache_8

    loss, softmax = nloss(Z, Y)
    
    return loss, softmax, model_cache
def backprop_step(softmax, Y, model_cache, params, alpha, adam_terms, t):
    B1 = 0.9
    B2 = 0.999
    epsilon = 10 ** -8
    #collect gradiants
    dZ = softmax - Y
    
    dA_prev, FC_dW2, FC_db2 = FC_backward(dZ, model_cache['8'])

    dZ = activation_backward(dA_prev, model_cache['7'], mode='tanh')

    dA_prev, FC_dW1, FC_db1 = FC_backward(dZ, model_cache['6'])
    
    dA_prev = dA_prev.reshape(-1, 4, 4, 16)
    
    dA_prev = pool_backward(dA_prev, model_cache['5'], mode='max')

    dZ = activation_backward(dA_prev, model_cache['4'], mode='tanh')
    
    dA_prev, C_dW2, C_db2 = conv_backward(dZ, model_cache['3'])

    dA_prev = pool_backward(dA_prev, model_cache['2'], mode='max')

    dZ = activation_backward(dA_prev, model_cache['1'], mode='tanh')

    dA_prev, C_dW1, C_db1 = conv_backward(dZ, model_cache['0'])
    

    #calculating terms for adam optimization
    adam_terms['VC_dW1'] = B1 * adam_terms['VC_dW1'] + (1 - B1) * C_dW1
    adam_terms['VC_db1'] = B1 * adam_terms['VC_db1'] + (1 - B1) * C_db1
    adam_terms['SC_dW1'] = B2 * adam_terms['SC_dW1'] + (1 - B2) * (C_dW1 ** 2)
    adam_terms['SC_db1'] = B2 * adam_terms['SC_db1'] + (1 - B2) * (C_db1 ** 2)
    adam_terms['VC_dW2'] = B1 * adam_terms['VC_dW2'] + (1 - B1) * C_dW2
    adam_terms['VC_db2'] = B1 * adam_terms['VC_db2'] + (1 - B1) * C_db2
    adam_terms['SC_dW2'] = B2 * adam_terms['SC_dW2'] + (1 - B2) * (C_dW2 ** 2)
    adam_terms['SC_db2'] = B2 * adam_terms['SC_db2'] + (1 - B2) * (C_db2 ** 2)
    
    adam_terms['VFC_dW1'] = B1 * adam_terms['VFC_dW1'] + (1 - B1) * FC_dW1
    adam_terms['VFC_db1'] = B1 * adam_terms['VFC_db1'] + (1 - B1) * FC_db1
    adam_terms['SFC_dW1'] = B2 * adam_terms['SFC_dW1'] + (1 - B2) * (FC_dW1 ** 2)
    adam_terms['SFC_db1'] = B2 * adam_terms['SFC_db1'] + (1 - B2) * (FC_db1 ** 2)
    adam_terms['VFC_dW2'] = B1 * adam_terms['VFC_dW2'] + (1 - B1) * FC_dW2
    adam_terms['VFC_db2'] = B1 * adam_terms['VFC_db2'] + (1 - B1) * FC_db2
    adam_terms['SFC_dW2'] = B2 * adam_terms['SFC_dW2'] + (1 - B2) * (FC_dW2 ** 2)
    adam_terms['SFC_db2'] = B2 * adam_terms['SFC_db2'] + (1 - B2) * (FC_db2 ** 2)
    
    #update parameters
    params['C_W1'] = params['C_W1'] - alpha * ((adam_terms['VC_dW1'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SC_dW1'] / (1 - (B2 ** t))) + epsilon))
    params['C_b1'] = params['C_b1'] - alpha * ((adam_terms['VC_db1'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SC_db1'] / (1 - (B2 ** t))) + epsilon))
    params['C_W2'] = params['C_W2'] - alpha * ((adam_terms['VC_dW2'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SC_dW2'] / (1 - (B2 ** t))) + epsilon))
    params['C_b2'] = params['C_b2'] - alpha * ((adam_terms['VC_db2'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SC_db2'] / (1 - (B2 ** t))) + epsilon))
    params['FC_W1'] = params['FC_W1'] - alpha * ((adam_terms['VFC_dW1'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SFC_dW1'] / (1 - (B2 ** t))) + epsilon))
    params['FC_b1'] = params['FC_b1'] - alpha * ((adam_terms['VFC_db1'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SFC_db1'] / (1 - (B2 ** t))) + epsilon))
    params['FC_W2'] = params['FC_W2'] - alpha * ((adam_terms['VFC_dW2'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SFC_dW2'] / (1 - (B2 ** t))) + epsilon))
    params['FC_b2'] = params['FC_b2'] - alpha * ((adam_terms['VFC_db2'] / (1 - (B1 ** t))) / np.sqrt((adam_terms['SFC_db2'] / (1 - (B2 ** t))) + epsilon))
    
    return params, adam_terms
df1 = sklearn.utils.shuffle(df1)
Y_train = df1.loc[:,'label'].to_numpy()
X_train = df1.loc[:,'1x1':].to_numpy().reshape(-1, 28, 28, 1)

#print(Y_train[3])
#image = X_train[3,:,:,0]
#fig = plt.figure
#plt.imshow(image, cmap='gray')
#plt.show()

df2 = sklearn.utils.shuffle(df2)
Y_test = df2.loc[:,'label'].to_numpy()
X_test = df2.loc[:,'1x1':].to_numpy().reshape(-1, 28, 28, 1)
params, hyperparam = initialize_parameters()
#train_set_X = X_train[:50,:,:,:]/255
#train_set_Y = Y_train[:50]
minibatch_size = 256
alpha = 0.005
adam_terms = {
    'VC_dW1': np.zeros_like(params['C_W1']), 'VC_db1':  np.zeros_like(params['C_b1']), 'VC_dW2':  np.zeros_like(params['C_W2']), 'VC_db2':  np.zeros_like(params['C_b2']),
    'VFC_dW1':  np.zeros_like(params['FC_W1']), 'VFC_db1':  np.zeros_like(params['FC_b1']), 'VFC_dW2':  np.zeros_like(params['FC_W2']), 'VFC_db2':  np.zeros_like(params['FC_b2']),
    'SC_dW1':  np.zeros_like(params['C_W1']), 'SC_db1':  np.zeros_like(params['C_b1']), 'SC_dW2':  np.zeros_like(params['C_W2']), 'SC_db2':  np.zeros_like(params['C_b2']),
    'SFC_dW1':  np.zeros_like(params['FC_W1']), 'SFC_db1':  np.zeros_like(params['FC_b1']), 'SFC_dW2':  np.zeros_like(params['FC_W2']), 'SFC_db2':  np.zeros_like(params['FC_b2'])
}
Y_set = np.zeros((10, Y_train.shape[0]))
count = 0

for x in Y_train:
    Y_set[int(x), count] = 1
    count += 1

for x in range(int(len(Y_train)/minibatch_size)):
    X = X_train[x * minibatch_size:(x+1)*minibatch_size,:,:,:] / 255
    Y = Y_set[:, x * minibatch_size:(x+1)*minibatch_size]
    
    loss, softmax, model_cache = forward_step(X, Y, params, hyperparam)

    params, adam_terms = backprop_step(softmax, Y, model_cache, params, alpha, adam_terms, x + 1)
    
    print ('Iteration loss:', x, loss)
correct = 0
minibatch_size = 256
for x in range(int(len(Y_test)/minibatch_size)):
    X = X_test[x * minibatch_size:(x+1)*minibatch_size,:,:,:] 
    Y = Y_test[x * minibatch_size:(x+1)*minibatch_size]
    
    loss, softmax, _ = forward_step(X, Y, params, hyperparam)
    
    predictions = np.argmax(softmax, axis=0)
    correct += np.sum(predictions == Y)
    accuracy = correct / ((x+1) * minibatch_size)
    print(accuracy)
x = np.asarray([[1, 2, 3],
                [4, 5, 6]])
print(np.sum(x, axis=0))
