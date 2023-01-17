# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/data.csv")
d = {'M':1,'B':0}
data = data.replace(d) #replace M/B with 0 or 1 for the neural net classification
data = data.drop(['Unnamed: 32'],axis=1) #remove column 32 - unknown purpose
data = data.drop(['id'],axis=1) #rem
data_temp = data.drop(['diagnosis'],axis=1) #make a temporary dataset with only our feature vectors
X = np.array(data_temp).T #create our Numpy array of feature vectors to be used in our neural net
Y = np.array(data['diagnosis']).T
X_mean = np.mean(X,axis=1,keepdims=True) #Find the mean of each feature
X_max = np.max(X,axis=1,keepdims=True) #Find the maximum of each feature
X_normalized = (X-X_mean)/(X_max) 

X_train = X_normalized[:,:380]
Y = Y.reshape(1,569)
X_train = X_normalized[:,:380]
Y_train = Y[:,:380]
Y_test = Y[:,381:]
class NNet(object):
    def __init__(self, X,Y,layer_dims, lr, iterations, dp = 1, out_act="sigmoid"):
        super(NNet,self).__init__()
        self.X = X
        self.Y = Y
        self.layer_dims = layer_dims 
        self.learning_rate = lr
        self.iterations = iterations
        self.keep_prob = dp
        self.out_act = out_act  

    def init_paras(self):
        L = len(self.layer_dims)  # number of layers in the network
        parameters = {}          
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l - 1])*0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))        
            assert(parameters['W' + str(l)].shape == (self.layer_dims[l], self.layer_dims[l - 1]))
            assert(parameters['b' + str(l)].shape == (self.layer_dims[l], 1))
            
        return parameters
    

    def sigmoid(self,z):
        s = 1 / (1 + np.exp(-z))
        cache = s    
        return s, cache # storing requried variables in cache as required for back propagation

    def softmax(self,z):
        s = np.exp(z)/np.sum(np.exp(z),axis=0)
        cache = s
        return s, cache

    # back functions are derivatives of respective activation
    def sig_back(self,dA, activation_cache):
        return dA*activation_cache*(1 - activation_cache)

    def soft_back(self,dA, activation_cache):

        ac = activation_cache
        dz = np.empty(ac.shape)

        for a in range(ac.shape[1]):
            s = ac[:,a].reshape(-1,1)
            ds = np.diagflat(s) - np.dot(s, s.T)
            dz[:,a] = np.matmul(ds,dA[:,a]) 
        
        assert(dz.shape == (dA.shape[0], dA.shape[1]))
        return dz

    def lin_forward(self,A, W, b,D):
        Z = np.dot(W, A) + b
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b, D)
        
        return Z, cache

        # applying activation after linear operation
    def lin_act_forward(self,A_prev,W,b,activation):

        if activation == "sigmoid":
            if np.array_equal(self.X,A_prev): 
                D = 0 # At start Drop Out should not apply to input layer as it is (X)   
                Z, linear_cache = self.lin_forward(A_prev, W, b,D)
                A, activation_cache = self.sigmoid(Z)
            else:
                D = np.random.rand(A_prev.shape[0], A_prev.shape[1])     
                D = D < self.keep_prob                            
                A_prev = A_prev * D                                 
                A_prev = A_prev / self.keep_prob 

                Z, linear_cache = self.lin_forward(A_prev, W, b,D)
                A, activation_cache = self.sigmoid(Z)

        elif activation == "softmax":
            D = np.random.rand(A_prev.shape[0], A_prev.shape[1])     
            D = D < self.keep_prob                             
            A_prev = A_prev * D                                     
            A_prev = A_prev / self.keep_prob

            Z, linear_cache = self.lin_forward(A_prev, W, b,D)
            A, activation_cache = self.softmax(Z)
        
        assert (A.shape == (W.shape[0], A_prev.shape[1]))
        cache = (linear_cache, activation_cache)

        return A, cache

    def forward_prop(self,in_X,parameters):
        caches = []
        A = in_X # input layer
        L = len(parameters) // 2             
        
        # hidden layers
        for l in range(1, L):
            A_prev = A 
            A, cache = self.lin_act_forward(A_prev,
                parameters['W' + str(l)],parameters['b' + str(l)],activation='sigmoid')

            caches.append(cache)
            
        W = parameters['W' + str(L)]
        b = parameters['b' + str(L)]

        # output layer
        if self.out_act == "softmax":
            AL, cache = self.lin_act_forward(A,W,b,activation='softmax')
        else:
            AL, cache = self.lin_act_forward(A,W,b,activation='sigmoid')

        caches.append(cache)
        
        assert(AL.shape == (self.Y.shape[0], in_X.shape[1]))
                
        return AL, caches

    def compute_cost(self,AL):
        m = Y.shape[1] # total number of training examples

        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        return cost


    def linear_backward(self,dZ, cache):

        A_prev, W, b, D = cache
        m = A_prev.shape[1]

        dW = np.dot(dZ, cache[0].T) / m
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True)) / m
        dA_prev = np.dot(cache[1].T, dZ)

        # backward dropout
        dA_prev = dA_prev*D
        dA_prev = dA_prev/self.keep_prob

        db = np.reshape(db,b.shape)
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        
        return dA_prev, dW, db



    def back_prop(self, AL, caches):
        grads = {}
        L = len(caches) 
        m = AL.shape[1]


        # Derivative of loss  
        dAL = - (np.divide(self.Y, AL) - np.divide(1 - self.Y, 1 - AL))
        current_cache = caches[-1]
        
        #output layer back prop.
        if self.out_act=="softmax":
            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(self.soft_back(dAL, 
                                                                                            current_cache[1]), 
                                                                                           current_cache[0])
        else:

            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(self.sig_back(dAL, 
                                                                                            current_cache[1]), 
                                                                                           current_cache[0])

        # hidden layer back prop.    
        for l in reversed(range(L-1)):
            
            current_cache = caches[l]
            dA = grads["dA" + str(l+2)]
            dA_prev_temp, dW_temp, db_temp = self.linear_backward(self.sig_back(dA, current_cache[1]), current_cache[0])
            grads["dA" + str(l + 1)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads


    def update_parameters(self, parameters,grads):
        L = len(parameters) // 2 

        for l in range(L):
            
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - self.learning_rate * grads["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - self.learning_rate * grads["db" + str(l + 1)]
            
        return parameters


    def train(self, print_cost=True): 
        costs = []                         
        
        parameters = self.init_paras()
        inx = self.X

        for i in range(0, self.iterations):
            AL, caches = self.forward_prop(inx,parameters)
            cost = self.compute_cost(AL)
            grads = self.back_prop(AL, caches)
            parameters = self.update_parameters(parameters,grads)
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" % (i, cost))
            if i % 100 == 0:
                costs.append(cost)

        train_y_pred = self.predict(inx, parameters) 
        
        print ("    Training accuracy:{}%" .format(self.accuracy(train_y_pred,self.Y)))

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.grid()
        plt.show()
        
        return parameters

    def predict(self,t_X,parameters):
        self.keep_prob = 1 # Dropout only used during training
        Y_pred, _ = self.forward_prop(t_X,parameters)
        Y_pred = Y_pred > 0.5  #np.argmax(Y_pred.T, axis=1)
        return Y_pred

    def accuracy(self,Y_pred,True_y):
        acc = Y_pred == True_y 

        #acc = acc.astype(np.float)
        acc = (np.sum(acc)*1.0/True_y.size)*100
        return acc
X = X_train
Y = Y_train
layer_dims=[30,16,1] # desired structure of NN with nodes in each layer

lr = 0.03 # learning rate
iterations = 20000
nn = NNet(X, Y, layer_dims, lr, iterations) 
para = nn.train()

X_test = X_normalized[:,381:]

y_predict = nn.predict(X_test,para)
print("Test Accuracy:{}%" .format(nn.accuracy(y_predict,Y_test)))
