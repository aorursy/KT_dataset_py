import numpy as np

import h5py

import matplotlib.pyplot as plt

import time

import scipy

from PIL import Image

from scipy import ndimage

%matplotlib inline

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'



def load_data():

    train_dataset = h5py.File('../input/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"])# your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:])# your train set labels



    test_dataset = h5py.File('../input/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

index = 10

#plt.imshow(train_set_x_orig[index])

#print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")





def preprocessing(train_input,test_input):

    train_x = train_x_orig.reshape(train_x_orig.shape[0],-1).T

    test_x = test_x_orig.reshape(test_x_orig.shape[0],-1).T

    train_x = train_x/255.0

    test_x = test_x/255.0

    return train_x,test_x



train_x,test_x = preprocessing(train_x_orig,test_x_orig)



def initialize_2_layer(n_x,n_h,n_o):

    np.random.seed(1)

    parameter = {}

    parameter['W1'] = np.random.randn(n_h,n_x)*0.01

    parameter['B1'] = np.zeros((n_h,1))

    parameter['W2'] = np.random.randn(n_o,n_h)*0.01

    parameter['B2'] = np.zeros((n_o,1))

    assert(parameter['W1'].shape == (n_h,n_x))

    assert(parameter['B1'].shape == (n_h,1))

    assert(parameter['W2'].shape == (n_o,n_h))

    assert(parameter['B2'].shape == (n_o,1))

    return parameter







def initialize_parameters_deep(layer_dims):

    np.random.seed(1)

    layer = len(layer_dims)

    parameter ={}

    

    for index in range(1,layer):

        parameter['W' + str(index)] = np.random.randn(layer_dims[index],layer_dims[index-1])*np.sqrt(2 / layers_dims[index - 1])

        parameter['B' + str(index)] = np.zeros((layer_dims[index],1))

        assert(parameter['W' + str(index)].shape == (layer_dims[index],layer_dims[index-1]))

        assert(parameter['B' + str(index)].shape == (layer_dims[index],1))

        

    return parameter



def sigmoid(Z):

    s = 1/(1+np.exp(-Z))

    return s,Z



def relu(Z):

    s = np.maximum(0,Z)

    return s,Z





def linear_forward(w,b,X):

    Z =  (np.dot(w,X)+b)

    return Z,(w,b,X)



def linear_activation_forward(w,b,X,activation):

    if(activation == 'relu'):

        Z,linear_cache = linear_forward(w,b,X)

        A,activation_cache = relu(Z)

    if(activation == 'sigmoid'):

        Z,linear_cache = linear_forward(w,b,X)

        A,activation_cache = sigmoid(Z)

    return A,(linear_cache,activation_cache) 





def linear_forwards(X,parameter):

    layer = int(len(parameter)/2)

    cache = []

    for index in range(1,layer):

     A = X

     X,temp_cache =linear_activation_forward(parameter['W'+str(index)],parameter['B'+str(index)],A,'relu')

     cache.append(temp_cache)

    AL,temp_cache =linear_activation_forward(parameter['W'+str(layer)],parameter['B'+str(layer)],X,'sigmoid') 

    cache.append(temp_cache) 

    return AL,cache

    

    

def computeCost(Al,Y):

    m = Y.shape[1]

    cost = -(1.0/m)*np.sum(np.multiply(Y,np.log(Al)+np.multiply((1-Y),np.log(1-Al))))

    return cost





def linear_backward(dZ,linear_cache):

    (w,b,X) = linear_cache

    m = dZ.shape[1]

    dW = (1.0/m)*np.dot(dZ,X.T)

    dB = (1.0/m)*np.sum(dZ,axis = 1,keepdims = True)

    dA = np.dot(w.T,dZ)

    return dW,dB,dA





def sigmoid_backward(activation_cache):

    s = sigmoid(activation_cache)[0]

    return s*(1-s)



def relu_backward(activation_cache):

    return 1*(activation_cache>0)



def linear_backward_function(dAl,activation,cache):

    linear_cache,activation_cache = cache

    if(activation == 'sigmoid'):

        dZ = dAl*sigmoid_backward(activation_cache)

        dW,dB,dA = linear_backward(dZ,linear_cache)

    if(activation == 'relu'):

        dZ = dAl*relu_backward(activation_cache)

        dW,dB,dA = linear_backward(dZ,linear_cache)

    return dW,dB,dA







def l_model_backward(cache,Al,Y):

    grad ={}

    m = Y.shape[1]

    layer = len(cache)

    dAl = -np.divide(Y,Al) + np.divide((1-Y),(1-Al))

    grad['dW'+str(layer)],grad['dB'+str(layer)],grad['dA'+str(layer-1)] = linear_backward_function(dAl,'sigmoid',cache.pop())

    for index in reversed(range(1,layer)):

        dAl = grad['dA' + str(index)]

        grad['dW'+str(index)],grad['dB'+str(index)],grad['dA'+str(index-1)] = linear_backward_function(dAl,'relu',cache.pop())

    return grad 







def updateParameter(parameter,grad,learning_rate):

    layer = int(len(parameter)/2)

    for index in range(1,layer+1):

        parameter['W' + str(index)] = parameter['W' + str(index)] - learning_rate*grad['dW' + str(index)]

        parameter['B' + str(index)] = parameter['B' + str(index)] - learning_rate*grad['dB' + str(index)]

    return parameter



def predict(X,Y,parameter):

    m = X.shape[1]

    p = np.zeros((1,m))

    proba,cache = linear_forwards(X,parameter)

    for index in range(0,m):

        if(proba[0][index] > 0.5):

            p[0][index] = 1

        else:

            p[0][index] = 0

    accuracy = (float(np.sum((Y == p)))/m)        

    print("accuracy is {}".format(accuracy))

    return p  





def print_mislabeled_images(classes,X,Y,p):

    a = Y+p

    mislabel_data = np.asarray(np.where(a == 1))

    plt.rcParams['figure.figsize'] = (60.0, 60.0) # set default size of plots

    image_size = len(mislabel_data[0])

    for index in range(0,image_size):

        index1 = mislabel_data[1][index]

        plt.subplot(2,image_size,index+1)

        plt.imshow(X[:,index1].reshape(64,64,3),interpolation='nearest')

        plt.axis('off')

        plt.title("Prediction:" + str(classes[int(p[0,index1])]))

    







    







def l2_model(X,Y,layer_dim,learning_rate,print_cost,iteration):

    parameter = {}

    grad= {}

    Al = 0

    costs = []

    np.random.seed(1)

    parameter = initialize_parameters_deep(layer_dim)

    for iter in range(0,iteration):

       Al,cache =  linear_forwards(X,parameter)

       cost = computeCost(Al,Y)

       grad =  l_model_backward(cache,Al,Y)

       parameter = updateParameter(parameter,grad,learning_rate)

       if((iter % 100) == 0 and print_cost == True):

            print("cost after {} iteration {}".format(iter,cost))

            costs.append(cost)

    plt.plot(costs)

    plt.xlabel("iteration in 100")

    plt.ylabel('cost')

    plt.title("learning rate" + str(learning_rate))

    plt.show()

    return parameter

    

n_x = 12288     # num_px * num_px * 3

n_h = 7

n_y = 1

layers_dims =  [12288, 20, 7,1]

    

parameter = l2_model(train_x,train_y,layers_dims,0.0075,True,5) 

p = predict(train_x,train_y,parameter)

p = predict(test_x,test_y,parameter)

##print_mislabeled_images(classes,test_x,test_y, p)





## START CODE HERE ##

my_image = "my_image.jpg" # change this to the name of your image file 

my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

## END CODE HERE ##



'''

fname = 'https://www.kaggle.com/sauravjalan/cat-image'

image = np.array(ndimage.imread(fname, flatten=False))

my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))

my_image = my_image/255.

'''
