import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
torch.set_default_tensor_type('torch.cuda.FloatTensor')
def initialize_parameters(layer_dims):

    L = len(layer_dims)
    parameters = {}
    print("Initializing parameters....")

    for l in range(1, L):
        parameters["W" + str(l)] = torch.cuda.FloatTensor(layer_dims[l], layer_dims[l-1]).normal_() * 0.01
        parameters["b" + str(l)] = torch.cuda.FloatTensor(layer_dims[l],1).normal_() * 0 + 1
        
        assert(parameters["W" + str(l)].shape == torch.Size([layer_dims[l],layer_dims[l-1]]))
        assert(parameters["b" + str(l)].shape == torch.Size([layer_dims[l],1]))
        
    print("Parameters Initialized....")

    for i in range(1, L):
        print("Shape of W{} : ".format(i), parameters["W" + str(i)].shape, "\t", parameters["W" + str(i)].type())

    return parameters
def initialize_trainingSet():
    
    #retrieving files...
    print("Retrieving Files....")
    
    train_X = pd.read_csv("../input/mnist_train.csv").drop(columns="5").values 
    train_Y = pd.read_csv("../input/mnist_train.csv").loc[:, "5"].values 
    test_X = pd.read_csv("../input/mnist_test.csv").drop(columns="7").values
    test_Y = pd.read_csv("../input/mnist_test.csv").loc[:, "7"].values
    
    print("Files Retrieved....")
    
    #reshaping the files into appropriate dimensions...
    
    train_X_orig = np.reshape(train_X, (train_X.shape[0], train_X.shape[1])).T/255
    train_Y_orig = np.reshape(train_Y, (train_Y.shape[0], 1)).T
    train_Y_orig = np.eye(10)[train_Y_orig].T
    train_Y_orig = np.reshape(train_Y_orig, (train_Y_orig.shape[0], train_Y_orig.shape[1] * train_Y_orig.shape[2]))
    
    test_X_orig = np.reshape(test_X, (test_X.shape[0], test_X.shape[1])).T/255
    test_Y_orig = np.reshape(test_Y, (test_Y.shape[0], 1)).T
    test_Y_orig = np.eye(10)[test_Y_orig].T
    test_Y_orig = np.reshape(test_Y_orig, (test_Y_orig.shape[0], test_Y_orig.shape[1] * test_Y_orig.shape[2]))
    
    train_X_orig = torch.from_numpy(train_X_orig).type(torch.cuda.FloatTensor)
    train_Y_orig = torch.from_numpy(train_Y_orig).type(torch.cuda.FloatTensor)
    test_X_orig = torch.from_numpy(test_X_orig).type(torch.cuda.FloatTensor)
    test_Y_orig = torch.from_numpy(test_Y_orig).type(torch.cuda.FloatTensor)
    
    return train_X_orig, train_Y_orig, test_X_orig, test_Y_orig
def suffle_training_data(train_X, train_Y, minibatch_size = 64):
    
    m = train_X.shape[1]
    complete_batch = int(m/minibatch_size)
    minibatches = []
    
    permutations = list(np.random.permutation(m))
    train_X = train_X[:, permutations]
    train_Y = train_Y[:, permutations]
    
    for k in range(complete_batch):
        
        mini_X = train_X[:, k * minibatch_size: (k + 1) * minibatch_size]
        mini_Y = train_Y[:, k * minibatch_size: (k + 1) * minibatch_size]
        minibatches.append((mini_X, mini_Y))
    
    if m%minibatch_size != 0:
        
        mini_X = train_X[:, complete_batch * minibatch_size:]
        mini_Y = train_Y[:, complete_batch * minibatch_size:]
        minibatches.append((mini_X, mini_Y))
    
    return minibatches
def non_linear_activation(Z, activation):
    
    if activation == "relu":
        zero = torch.cuda.FloatTensor([0])
        Z = torch.max(zero.expand_as(Z), Z)
    elif activation == "softmax":
        Z = torch.exp(Z)
        Z_sum = torch.sum(Z, dim=0)
        Z = Z/Z_sum
    return Z
def forward_propagation(X, parameters, layer_dims):
    
    L = len(layer_dims)
    cache = {}
    cache["A0"] = X
    A = X
    
    for l in range(1, L):
        cache["Z" + str(l)] = torch.mm(parameters["W" + str(l)], A) + parameters["b" + str(l)]
        cache["A" + str(l)] = non_linear_activation(cache["Z" + str(l)], activation = "relu")

        assert(cache["Z" + str(l)].shape == cache["A" + str(l)].shape)
        A = cache["A" + str(l)]
        
    cache["A" + str(L - 1)] = non_linear_activation(cache["Z" + str(L - 1)], activation = "softmax")
    assert(cache["Z" + str(L - 1)].shape == cache["A" + str(L - 1)].shape)
    
    return cache
def compute_cost(Y, AL, layer_dims, parameters, lmbd = 0.07):

    '''
    Shape of Y :  (no. of output units, no. of examples)
    Shape of AL :  (no. of output units, no. of examples)
    '''
    m = int(Y.shape[1])
    L = len(layer_dims)
    
    regularized_term = 0
    for l in range(1, L):
        regularized_term = regularized_term + torch.sum(torch.mul(parameters["W" + str(l)], parameters["W" + str(l)]))
    
    cost = (-1/m) * torch.sum(torch.mul(Y, torch.log(AL)) + torch.mul((1 - Y), torch.log(1 - AL))) + (lmbd * regularized_term)/(2 * m)  
    
    return cost
def relu_derivative(Z):
    Z[Z <= 0] = 0
    Z[Z > 0] = 1
    return Z
def backpropagate(parameters, cache, layer_dims, Y, lmbd = 0.07):

    grads = {}
    L = len(layer_dims) - 1
    
    m = int(Y.shape[1])
    
    grads["dZ{}".format(L)] = (cache["A{}".format(L)] - Y)
    grads["dW{}".format(L)] = (1/m) * torch.mm(grads["dZ{}".format(L)], cache["A{}".format(L - 1)].permute(1,0))
    grads["db{}".format(L)] = (1/m) * torch.sum(grads["dZ{}".format(L)], dim=1, keepdim=True)
    
    W = parameters["W{}".format(L)]
    dZ = grads["dZ{}".format(L)]
    for l in range(1, L):
        
        grads["dA" + str(L - l)] = torch.mm(W.permute(1,0), dZ) 
        grads["dZ" + str(L - l)] = grads["dA" + str(L - l)] * relu_derivative(cache["Z" + str(L - l)])
        grads["dW" + str(L - l)] = ((1/m) * torch.mm(grads["dZ" + str(L - l)], cache["A" + str(L - l - 1)].permute(1,0))) + ((lmbd * parameters["W" + str(L - l)])/m)
        grads["db" + str(L - l)] = (1/m) * torch.sum(grads["dZ" + str(L - l)], dim=1, keepdim=True)
        
        assert(grads["dA" + str(L - l)].shape == cache["A" + str(L - l)].shape)
        assert(grads["dZ" + str(L - l)].shape == cache["Z" + str(L - l)].shape)
        assert(grads["dW" + str(L - l)].shape == parameters["W" + str(L - l)].shape)
        assert(grads["db" + str(L - l)].shape == parameters["b" + str(L - l)].shape)
        
        W = parameters["W" + str(L - l)]
        dZ = grads["dZ" + str(L - l)]

    return grads
def init_adam(parameters, L):
    
    v = {}
    s = {}
    
    for l in range(1,L):
        v["dW" + str(l)] = torch.from_numpy(np.zeros_like(parameters["W" + str(l)])).type(torch.cuda.FloatTensor)
        v["db" + str(l)] = torch.from_numpy(np.zeros_like(parameters["b" + str(l)])).type(torch.cuda.FloatTensor)
        
        s["dW" + str(l)] = torch.from_numpy(np.zeros_like(parameters["W" + str(l)])).type(torch.cuda.FloatTensor)
        s["db" + str(l)] = torch.from_numpy(np.zeros_like(parameters["b" + str(l)])).type(torch.cuda.FloatTensor)
    
    return v,s
def update_parameters(grads, parameters, v, s, t, learning_rate = 0.03, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    
    L = len(layer_dims)
    v_corrected = {}
    s_corrected = {}

    for l in range(1,L):
        
        v["dW" + str(l)] = (beta1 * v["dW" + str(l)]) + ((1 - beta1) * grads["dW" + str(l)])
        v["db" + str(l)] = (beta1 * v["db" + str(l)]) + ((1 - beta1) * grads["db" + str(l)])
        
        s["dW" + str(l)] = (beta2 * s["dW" + str(l)]) + ((1 - beta2) * torch.pow(grads["dW" + str(l)], 2))
        s["db" + str(l)] = (beta2 * s["db" + str(l)]) + ((1 - beta2) * torch.pow(grads["db" + str(l)], 2))
        
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - torch.pow(torch.cuda.FloatTensor([beta1]), t))
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - torch.pow(torch.cuda.FloatTensor([beta1]), t))
        
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - torch.pow(torch.cuda.FloatTensor([beta2]), t))
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - torch.pow(torch.cuda.FloatTensor([beta2]), t))
        
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * (v_corrected["dW" + str(l)]/ (torch.sqrt(s_corrected["dW" + str(l)]) + epsilon)))
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * (v_corrected["db" + str(l)]/ (torch.sqrt(s_corrected["db" + str(l)]) + epsilon)))

    return parameters, v, s
def model(train_X, train_Y, layer_dims, learning_rate, beta1, beta2, lmbd, epochs = 10, minibatch_size = 64, print_cost=False):
    
    m = train_X.shape[1]
    costs = []
    t = 0
    L = len(layer_dims)
    parameters = initialize_parameters(layer_dims)
    v,s = init_adam(parameters, L)
    
    minibatches = suffle_training_data(train_X, train_Y, minibatch_size)
    counter = 0
    
    for epoch in range(epochs):
        
        for minibatch in minibatches :
            
            (mini_X, mini_Y) = minibatch
            counter += 1            
            cache = forward_propagation(mini_X, parameters, layer_dims)
            AL = cache["A" + str(L - 1)]
            cost = compute_cost(mini_Y, AL, layer_dims, parameters, lmbd)
            grads = backpropagate(parameters, cache, layer_dims, mini_Y, lmbd)
            t = t + 1
            parameters, v, s = update_parameters(grads, parameters, v, s, t, learning_rate, beta1, beta2, 1e-8)
            costs.append(cost)
            
        counter = 0
        if print_cost and epoch % 1 == 0:
            print("Cost after " + str(epoch) + " epochs : " + str(cost))
    
    plt.plot(costs)
    plt.xlabel("No. of Iteration")
    plt.ylabel("cost")
    plt.show()
    
    cache = forward_propagation(train_X, parameters, layer_dims)
    AL = cache["A" + str(L - 1)]
    AL[AL <= 0.5] = 0
    AL[AL > 0.5] = 1
    diff = (torch.sum(torch.abs(train_Y - AL))/(m)) * 100
    train_accuracy = 100 - diff
    
    return parameters, train_accuracy, costs
def prediction_accuracy(test_X, test_Y, parameters, layer_dims):
    L = len(layer_dims)
    m = test_Y.shape[1]
    pred_cache = forward_propagation(test_X, parameters, layer_dims)
    pred_cache["A{}".format(L - 1)][pred_cache["A{}".format(L - 1)] <= 0.5] = 0
    pred_cache["A{}".format(L - 1)][pred_cache["A{}".format(L - 1)] > 0.5] = 1
    diff = (torch.sum(torch.abs(test_Y - pred_cache["A{}".format(L - 1)]))/m) * 100
    test_accuracy = 100 - diff
    
    return test_accuracy
def digit_recognizer(test_X, layer_dims):

    newCache = {}
    L = len(layer_dims)
    optimized_parameters = pickle.load(open("trained_parameters.pickle", "rb"))
    
    while True:
        index = int(input("Enter any example number : "))
        if index == -1:
            break
        X = test_X[:, index].view(784, 1)
        print(X.shape)
        for l in range(1, L):
            newCache["Z" + str(l)] = torch.mm(optimized_parameters["W" + str(l)], X) + optimized_parameters["b" + str(l)]
            newCache["A" + str(l)] = non_linear_activation(newCache["Z" + str(l)], activation = "relu")
            
            assert(newCache["Z" + str(l)].shape == newCache["A" + str(l)].shape)
            X = newCache["A" + str(l)]
        
        newCache["A{}".format(L - 1)] = non_linear_activation(newCache["Z{}".format(L - 1)], activation = "softmax")
        AL = newCache["A{}".format(L - 1)]
        assert(newCache["Z{}".format(L - 1)].shape == newCache["A{}".format(L - 1)].shape)
        
        temp = test_X[:, index].contiguous()
        img = temp.view(28,-1)
        AL = AL.cpu().numpy()
        print("Label : ", list(AL).index(max(list(AL))))
        print("Probability distribution : ", AL)

        plt.imshow(img)
        plt.show()
train_X_orig, train_Y_orig, test_X_orig, test_Y_orig = initialize_trainingSet()
layer_dims = [784, 100, 100, 50, 10]
optimized_parameters, train_accuracy, costs = model(train_X_orig, train_Y_orig, layer_dims, learning_rate = 0.00038, beta1 = 0.9, beta2 = 0.999, lmbd = 0.05, epochs = 35, minibatch_size = 128, print_cost=True)
test_accuracy = prediction_accuracy(test_X_orig, test_Y_orig, optimized_parameters, layer_dims)

print("\n\nWriting paramters to the disk....")
optimized_params = open("trained_parameters.pickle", "wb")
pickle.dump(optimized_parameters, optimized_params)
print("Paramters Written to disk....", "\n\n")

print("Train accuracy : ", train_accuracy)
print("Test accuracy : ", test_accuracy)

new_optimized_parameters = optimized_parameters
