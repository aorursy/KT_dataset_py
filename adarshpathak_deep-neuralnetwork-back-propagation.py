import numpy as np

from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings('ignore')
class Utils:

    @staticmethod

    def sigmoid(Z): 

        A = 1/(1+np.exp(-Z))

        cache = Z

        return A, cache

    

    @staticmethod

    def relu(Z):

        A = np.maximum(0,Z)

        cache = Z 

        return A, cache

    

    @staticmethod

    def relu_backward(dA, cache):

        Z = cache

        dZ = np.array(dA, copy=True)

        dZ[Z <= 0] = 0

        return dZ

    

    @staticmethod

    def sigmoid_backward(dA, cache):

        Z = cache

        s = 1/(1+np.exp(-Z))

        dZ = dA * s * (1-s) 

        return dZ
class DNN(Utils):

    

    '''

    input:

        X: input features

        y: target

        L: (array-like) number of hidden layers with units 

        lr: learning rate

        

        example:

        import numpy as np

        features = np.random.randn(100,8)

        targets = np.random.randn(100,1)

        L = [16,32,32,64]

        lr = 1e-3

        model = DNN(X = features,y=targets,layers=L,learning_rate=lr)

    '''

    

    def __init__(self,X,y,layers,learning_rate):

        super().__init__()

        self.x = X

        self.y = y

        self.L = layers

        self.lr = learning_rate

    

    def initialize_parameters(self):

        layer = len(self.L)

        n = self.x.shape[1]

        parameters = {}

        for l in range(1,layer):

            parameters["W"+str(l)] = np.random.randn(self.L[l],self.L[l-1]) * 1/np.sqrt(n ** (layer-1))

            parameters["b"+str(l)] = np.zeros((self.L[l],1))

        

        return parameters

    

    def linear_forward(self,A,W,b):

        Z = np.dot(W,A) + b

        cache = (A,W,b)

        return Z,cache

    

    def linear_activation_forward(self,A_prev,W,b,activation):

        if activation == "sigmoid":

            Z, linear_cache = self.linear_forward(A_prev, W, b)

            A,activation_cache = Utils.sigmoid(Z)

        elif activation == "relu":

            Z, linear_cache = self.linear_forward(A_prev, W, b)

            A,activation_cache = Utils.relu(Z)

        

        cache = (linear_cache,activation_cache)

        return A,cache

    

    def forward(self,parameters):

        caches = []

        A = self.x

        L = len(parameters)//2

        

        for l in range(1,L):

            A_prev = A

            A,cache = self.linear_activation_forward(A_prev,

                                               parameters["W"+str(l)],

                                               parameters["b"+str(l)],

                                               activation = "relu")

            caches.append(cache)

        

        AL,cache = self.linear_activation_forward(A,

                                             parameters["W"+str(L)],

                                            parameters["b"+str(L)],

                                            activation = "sigmoid")

        caches.append(cache)

        

        return AL,caches

    

    def compute_cost(self,AL):

        Y = self.y

        m = Y.shape[1]

        cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))

        cost = np.squeeze(cost)

        return cost

    

    def linear_backward(self,dZ,cache):

        A_prev, W, b = cache

        m = A_prev.shape[1]

        dW = 1/m*np.dot(dZ ,A_prev.T)

        db = 1/m*np.sum(dZ , axis = 1, keepdims=True)

        dA_prev = np.dot(W.T ,dZ)

        return dA_prev, dW, db

    

    def linear_activation_backward(dA, cache, activation):

        linear_cache, activation_cache = cache

        if activation == "relu":

            dZ = Utils.relu_backward(dA, activation_cache)

            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":

            dZ = Utils.sigmoid_backward(dA, activation_cache)

            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    

    def backward(self,AL,caches):

        Y = self.y

        grads = {}

        m = AL.shape[1]

        L = len(caches)

        Y = Y.reshape(AL.shape)

        

        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[-1]

        grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_backward(Utils.sigmoid_backward(dAL, 

                                                                                                        current_cache[1]), 

                                                                                                        current_cache[0])

        for l in reversed(range(L-1)):

            current_cache = caches[l]

            dA_prev_temp, dW_temp, db_temp = self.linear_backward(Utils.sigmoid_backward(dAL,

                                                                              current_cache[1]),

                                                             current_cache[0])

            grads["dA" + str(l + 1)] = dA_prev_temp

            grads["dW" + str(l + 1)] = dW_temp

            grads["db" + str(l + 1)] = db_temp

            

        return grads

    

    def update_parameters(self,parameters,grads):

        learning_rate = self.lr

        L = len(parameters) // 2

        for l in range(L):

            parameters["W" + str(l+1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]

            parameters["b" + str(l+1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

        return parameters

    

    def training(self,iterations=100,print_cost=False,print_interval=10):

        parameters = self.initialize_parameters()

        costs = []

        for i in tqdm_notebook(range(0,iterations)):

            AL,caches = self.forward(parameters)

            cost = self.compute_cost(AL)

            grads = self.backward(AL,caches)

            parameters = self.update_parameters(parameters,grads)

            if print_cost & i % print_interval:

                print(f'cost after interval {i}: {cost}')

                costs.append(cost)

        return costs