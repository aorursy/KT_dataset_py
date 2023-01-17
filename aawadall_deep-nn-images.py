# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt # Visualization

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Load Data

train_raw = pd.read_csv("../input/train.csv")

test_raw = pd.read_csv("../input/test.csv")

print("Data Loaded")
def get_features(raw_data):

    cols = []

    # Get data of each row from pixel0 to pixel783 

    for px in range(784):

        cols.append("pixel"+str(px))   

    return raw_data.as_matrix(cols)/255



def cross_validated(X, n_samples):

    kf = KFold(n_samples, shuffle = True)

    result = [group for group in kf.split(X)]

    return result        
# Deep Implementation 

# Initialize Parameters 

def init_dnn_parameters(n, activations, epsilons, filter=None):

    L = len(n)

    params = {}

    for l in range(1,L):

        W = np.random.randn(n[l],n[l-1]) * epsilons[l] 

        # Experiment, multiply filter in case of input layer weights 

        if filter1 is not None and l == 1:

            W = np.dot(W, filter) 

        b = np.zeros((n[l],1))

        params["W"+str(l)] = W

        #print(" initizlizing W"+str(l))

        params["b"+str(l)] = b                        

        params["act"+str(l)] = activations[l]

    params["n"] = n

    return params



# Activation Functions 

def gdnn(X, activation_function):

    leak_factor = 1/10000

    if activation_function == 'tanh':

        return np.tanh(X)

    if activation_function == 'lReLU':

        return ((X > 0) * X) + ((X <= 0)* X * leak_factor)

    else: 

        return 1 / (1 +np.exp(-X))



def gdnn_prime(X, activation_function):

    leak_factor = 1/10000

    if activation_function == 'tanh':

        return 1-np.power(X,2)

    if activation_function == 'lReLU':

        return ((X > 0) * 1) + ((X <= 0)* leak_factor)

    else: 

        return (1 / (1 +np.exp(-X)))*(1-(1 / (1 +np.exp(-X))))



# Cost 

def get_dnn_cost(Y_hat, Y):

    #print(Y.shape)

    m = Y.shape[1]

    logprobs = np.multiply(np.log(Y_hat),Y) + np.multiply(np.log(1-Y_hat),1-Y)

    cost = - np.sum(logprobs) /m

    return cost

    

# Forward Propagation 

def forward_dnn_propagation(X, params):

    # Get Network Parameters 

    n = params["n"]

    L = len(n)

    

    A_prev = X

    cache = {}

    cache["A"+str(0)] = X

    for l in range(1,L):

        W = params["W"+str(l)]

        b = params["b"+str(l)]

        Z = np.dot(W,A_prev)+b

        A = gdnn(Z,params['act'+str(l)])

        cache["Z"+str(l)] = Z

        cache["A"+str(l)] = A

        

        A_prev = A

    return A, cache 



# Backward Propagation

def back_dnn_propagation(X, Y, params, cache, alpha = 0.01, _lambda=0, keep_prob=1):

    n = params["n"]

    L = len(n) -1

    

    m = X.shape[1]

    W_limit = 5

    A = cache["A"+str(L)]

    A1 = cache["A"+str(L-1)]

    #print("back_dnn_propagation: A(L) shape"+str(A.shape))

    #print("back_dnn_propagation: A1(L) shape"+str(A1.shape))

    grads = {}

    

    # Outer Layer 

    dZ = A - Y#gdnn_prime(A - Y, params["act"+str(L)])

    #print("back_dnn_propagation: dZ(L) shape"+str(dZ.shape))

    dW = 1/m * np.dot(dZ, A1.T)

    #print("back_dnn_propagation: dW(L) shape"+str(dW.shape))

    db = 1/m * np.sum(dZ, axis=1, keepdims=True)

    grads["dZ"+str(L)] = dZ

    grads["dW"+str(L)] = dW + _lambda/m * params["W"+str(L)]

    grads["db"+str(L)] = db

    

    # Update Outer Layer

    params["W"+str(L)] -= alpha * dW

    #params["W"+str(L)] = np.clip(params["W"+str(L)],-W_limit,W_limit)

    params["b"+str(L)] -= alpha * db

    for l in reversed(range(1,L)):

        #dZ2 = A2 - Y

        #dW2 = 1/m * np.dot(dZ2, A1.T)

        #db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

        #dZ1 = np.dot(W2.T, dZ2)*g_prime(A1)

        #dW1 = 1/m * np.dot(dZ1, X.T)

        #db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

        

        dZ2 = dZ

        W2 = params["W"+str(l+1)]

        b = params["b"+str(l)]

        A2 = cache["A"+str(l)]

        A1 = cache["A"+str(l-1)]

        d = np.random.randn(A1.shape[0],A1.shape[1]) > keep_prob

        A1 = A1 * d/keep_prob

        dZ = np.dot(W2.T, dZ2)*gdnn_prime(A2, params["act"+str(l)])



        dW = 1/m * np.dot(dZ, A1.T) + _lambda/m * params["W"+str(l)]

        

        db = 1/m * np.sum(dZ, axis=1, keepdims=True)

        grads["dZ"+str(l)] = dZ

        grads["dW"+str(l)] = dW

        grads["db"+str(l)] = db

        params["W"+str(l)] -= alpha *dW

        #params["W"+str(l)] = np.clip(params["W"+str(l)],-W_limit,W_limit)

        params["b"+str(l)] -= alpha *db

    

    return grads, params    



def batch_back_propagation(X, Y, params, cache, alpha = 0.01, _lambda=0, keep_prob=1,batch_size=128):

    # slice input and output data into smaller chunks 

    m = X.shape[1]

    include_probability = keep_prob

    idx_from = 0

    idx_to = min(batch_size, m)

    X_train = X[:,idx_from:idx_to]

    y_train = Y[:,idx_from:idx_to]

    while idx_to < m:

        #print("Epoch from {} to {}".format(idx_from, idx_to))

        if np.random.random() < include_probability:

            A, cache = forward_dnn_propagation(X_train, params)

            grads, params= back_dnn_propagation(X_train, y_train, params, cache, alpha ,_lambda, keep_prob)    

        idx_from += batch_size

        idx_from = min(m, idx_from)

        idx_to += batch_size

        idx_to = min(m, idx_to)

        if idx_from < idx_to:

            X_train = X[:,idx_from:idx_to]

            y_train = Y[:,idx_from:idx_to]

    return grads, params
X2 = get_features(train_raw)



labels = np.array(train_raw['label'])

#labels = np.reshape(labels,(-1,1))

m = labels.shape[0]

y = np.zeros((m,10))

for j in range(10):

    y[:,j]=(labels==j)*1



k = 38

folds = 3

oinst = 1

h_layers = 5

np.random.seed(1)

cv_groups = cross_validated(X2, folds)



alphas = np.linspace(0.045, 0.045, oinst)

epsilons = np.linspace(0.76,0.78,oinst)

gammas =  np.linspace(0.01,0.01,oinst)

lambdas=  np.linspace(25.91,25.91,oinst)

keep_probs=  np.linspace(0.9,0.9,oinst)

alph_decays = np.linspace(0.995,0.995,oinst) 

n_1 = []

break_tol = 0.00000001

etscost = []

etrcost= []

seeds = []

layers = []

for j in range(oinst):

    batch_processing = True

    batch_size = 1024



    #X = kernel(X2,X2,y,k,epsilon) # RBF

    X = X2 # Direct Map

    n = [X.shape[1]]

    acts = ['input']

    gamma = [0]

    for layer in range(h_layers):

        #n.append((10+h_layers-layer)**2) #((28-layer*3))**2)

        n.append((14-layer)**2) #((28-layer*3))**2)

        acts.append('lReLU') #tanh')

        #gamma.append(gammas[j]/(2**(h_layers-layer)))

        gamma.append(np.sqrt(2/n[layer-1]))

    layers.append(j+1)    

    n.append(y.shape[1])

    acts.append('sigmoid')

    #gamma.append(gammas[j])

    gamma.append(np.sqrt(1/n[layer-1]))

    #wall = 23

    n_1.append(j+4)

    np.random.seed(1)

    iterations = 150

    

    alpha = alphas[j]#0.166# 

    

    _lambda = lambdas[j] # 0.5#

    keep_prob = keep_probs[j]

    epsilon = 0.76#epsilons[j] #0.02 

    

    #n = [X.shape[1],wall,wall-1,wall-2,n_1[-1] ,16,1]

    L = len(n) - 1

    #acts = ['x','lReLU', 'lReLU',  'lReLU', 'lReLU', 'lReLU','sigmoid']



    # Prepare Training and testing sets 

    X_train = X[cv_groups[0][0],:].T 

    y_train = y[cv_groups[0][0],:].T 

    labels_train = labels[cv_groups[0][0]]

    # Experiment - Filter based on linear correlation 

    depth = 1024

    filter1 = np.zeros((n[0],n[0]))

    for dim in range(10):

        for monomial in range(1,min(3, h_layers)):

            X_sample = X_train[:,:depth].T**monomial

            X_mean = np.reshape(np.mean(X_sample,axis=0),(1,-1))

            y_sample = np.reshape(y_train[dim, :depth],(-1,1))



            y_mean = np.mean(y_sample)

            y_var = (y_sample - y_mean)*X_sample**0

            numer = (np.dot((X_sample-X_mean).T,y_var))

            denom = np.sqrt(np.sum(np.dot((X_sample-X_mean).T,(X_sample-X_mean))))*np.sqrt(np.dot((y_sample - y_mean).T,(y_sample - y_mean)))

            filter1 += np.diag((numer/denom)[:,0])

    filter1 /= np.linalg.norm(filter1)

    filter2 = 1*(np.abs(filter1) > 0.0001 )

    params = init_dnn_parameters(n, acts,gamma, filter2)



    # Experiment 

    

    X_test = X[cv_groups[0][1],:].T 

    y_test = y[cv_groups[0][1],:].T

    print("Exp[{}] - Eps = {}, Alph = {}, Decay = {}, lambda={}".format(j, epsilon, alpha,alph_decays[j], _lambda))

    print("k = {}, |X| = {}, max(i) = {}".format( k, X_test.shape[0], iterations))

    print("Keep Prob = {}, gamma = {}".format(keep_prob, gamma))

    print("Network {} {}".format(n,acts))

    cost = []

    tcost=[]

    A, cache = forward_dnn_propagation(X_train, params)

    for i in range(iterations):

        if batch_processing:

            grads, params = batch_back_propagation(X_train, 

                                                   y_train, 

                                                   params, 

                                                   cache, 

                                                   alpha,

                                                   _lambda, 

                                                   keep_prob,                                                  

                                                   batch_size)

            A, cache = forward_dnn_propagation(X_train, params)

            cost.append(np.mean(get_dnn_cost(A, y_train)))

            A2, vectors = forward_dnn_propagation(X_test, params)

            tcost.append(get_dnn_cost(A2, y_test))

        else:

            A, cache = forward_dnn_propagation(X_train, params)

            cost.append(get_dnn_cost(A, y_train))

            grads, params= back_dnn_propagation(X_train, y_train, params, cache, alpha ,_lambda, keep_prob)

            A2, vectors = forward_dnn_propagation(X_test, params)

            tcost.append(get_dnn_cost(A2, y_test))

        

        if alpha*np.abs(np.linalg.norm(grads["dW"+str(L)])) < break_tol:

            break

        if i % 10 == 0:

            alpha *= alph_decays[j]

            print("---------------------------------------------------------------")

            print("i = {}, trc = {}, tsc={}, alph.dWL = {}".format(i,cost[-1],

                                                                   tcost[-1], 

                                                                   alpha*np.abs(np.linalg.norm(grads["dW"+str(L)]))))

            print(" active alph = {}".format(alpha))

            print("Number Matching")

            for num in range(10):

                y_hat = A2[num,:] > 0.5

                y_star = y_test[num,:]

                matched = np.sum((1-np.abs(y_star-y_hat))*y_star)

                distance = np.linalg.norm((y_star - A2[num,:])*y_star)

                m_test = sum(y_test[num,:]==1)

                y_size = y_test.shape[1]

                pct = matched/m_test

                print("[{}] Matched {} {}% m_pos={}, Distance {}".format(num, matched,pct*100, m_test,distance ))

    etscost.append(tcost[-1])

    etrcost.append(cost[-1])

    plt.plot(tcost, label="testing case"+str(j))

    plt.plot(cost, '--',label="training case"+str(j))

    plt.legend()


#print(X.shape)

#print(params["W1"].shape)

#print(params["act1"])

L = len(n) 

#print(params["act"+str(L-1)])

if 1 ==1:

    

    plt.imshow(np.reshape(np.diag(filter2),(28,28)))

    plt.colorbar()

    plt.show()



    for l in range(10):

        im = X[l,:]

        side = int(np.sqrt(im.shape[0]))

        im = np.reshape(im,(side, side))

        plt.figure()

        plt.imshow(im, cmap='hot')

        plt.show()

    

#for l in range(1,L):

#if 1==1:

    for l in range(1,min(10,L)):

    

        plt.imshow(params["W"+str(l)])

        plt.colorbar()

        for node in range(min(25,params["W"+str(l)].shape[0]-1)):

            print(node, params["W"+str(l)].shape[1])

            W_node = params["W"+str(l)][node]

            side = int(np.sqrt(W_node.shape[0]))

            W_node = np.reshape(W_node, (side,side))        

            plt.figure()

            im = plt.imshow(W_node, cmap='hot')   

            #CS = plt.contour(W_node, cmap='gray')

            #plt.clabel(CS, inline=5, fontsize=10)

            plt.colorbar(im) 

            plt.title("Weight Vector of Node {} in Layer {}".format(node, l))



            plt.show()
if 1==0:

    print(y_test.shape)

    print(A2[:,0:10]>0.5)

    print(X2.shape)

    plt.scatter(A2[:,0],y_test[:,0])
X_test = get_features(test_raw)

#test_raw.describe(include="all")
A2, vectors = forward_dnn_propagation(X_test.T, params)

# use A2 to construct 



y_hat = A2 > 0.5

y_hat = y_hat*1



data = np.zeros((y_hat.shape[1],1))

for idx in range(y_hat.shape[0]):

    data += idx * np.reshape(y_hat[idx,:],(-1,1))

    

#print(data)

s1 = pd.Series(data[:,0])



df = pd.DataFrame(data = s1)



df.columns = ['Label']

df.to_csv('deep_nn.csv', sep=',')
#print(X[0:1024,:].T.shape)

#print(np.reshape(labels[:1024],(1,-1))*X[0:1024,:].T**0)

#print(np.corrcoef(X[0:1024,0].T, np.reshape(labels[0:1024],(1,-1))))
