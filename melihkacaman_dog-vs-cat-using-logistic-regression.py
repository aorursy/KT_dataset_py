
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 

import warnings 
# Filter Warnings 
warnings.filterwarnings('ignore')

from PIL import Image
import os

PATH_TRAIN_CAT = '/kaggle/input/dogs-cats-images/dataset/training_set/cats' 
PATH_TRAIN_DOG = '/kaggle/input/dogs-cats-images/dataset/training_set/dogs' 
PATH_TEST_CAT = '/kaggle/input/dogs-cats-images/dataset/test_set/cats' 
PATH_TEST_DOG = '/kaggle/input/dogs-cats-images/dataset/test_set/dogs' 

CAT = 1 
DOG = 0

IMG_WIDTH = 200 
IMG_HEIGHT = 200 
NR_PIX = IMG_HEIGHT * IMG_WIDTH
NR_CHANELS = 3
NR_FEATURES = NR_PIX * NR_CHANELS
def load_data_asanarray(path):
    data = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            specific_path = os.path.join(dirname, filename)
            image = Image.open(specific_path).resize((200,200))
            img_array = np.asarray(image)
            data.append(img_array)
    return np.asarray(data).reshape(len(data),IMG_WIDTH, IMG_HEIGHT, NR_CHANELS)
%%time
data_train_cat = load_data_asanarray(PATH_TRAIN_CAT)
data_train_dog = load_data_asanarray(PATH_TRAIN_DOG)
print(f'data_train_cat size is {len(data_train_cat)} data_train_dog size is {len(data_train_dog)}')
y_label_cat = np.ones((len(data_train_cat),1))
y_label_dog = np.zeros((len(data_train_dog),1))
x_train = np.concatenate((data_train_cat, data_train_dog), axis=0)
y_train = np.concatenate((y_label_cat, y_label_dog), axis=0)
%%time
data_test_cat = load_data_asanarray(PATH_TEST_CAT)
data_test_dog = load_data_asanarray(PATH_TEST_DOG)
print(f'data_train_cat size is {len(data_train_cat)} data_train_dog size is {len(data_train_dog)}')
y_label_cat = np.ones((len(data_test_cat),1))
y_label_dog = np.zeros((len(data_test_dog),1))
x_test = np.concatenate((data_test_cat, data_test_dog), axis=0)
y_test = np.concatenate((y_label_cat, y_label_dog), axis=0)
print(f'x_test shape is {x_test.shape} and y_shape is {y_test.shape}')
X = np.concatenate((x_train, x_test), axis=0)
Y = np.concatenate((y_train, y_test), axis=0) 

x_train,x_test,y_train, y_test = train_test_split(X,Y, test_size=0.15, random_state=45)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
x_train = x_train.reshape(len(x_train), NR_FEATURES).T
x_test = x_test.reshape(len(x_test), NR_FEATURES).T
y_train = y_train.T 
y_test = y_test.T 
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
def init_param(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0 
    
    return w,b 
def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    sig = np.minimum(sig, 0.9999)  # Set upper bound
    sig = np.maximum(sig, 0.0001)  # Set lower bound
    return sig
x_train, x_test = x_train / 255, x_test/ 255 
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b 
    y_head = sigmoid(z)
    
    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    return cost 
def forward_and_backward_propagation(w,b,x_train, y_train):
    z = np.dot(w.T, x_train) + b 
    y_head = sigmoid(z) 
        
    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]  
   
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost, gradients
def train(w,b,x_train,y_train,learning_rate, nr_iteration):
    cost_list = [] 
    cost_list2 = [] 
    index = [] 
    
    for i in range(nr_iteration):
        cost, gradients = forward_and_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        
        if i%10 == 0 :
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
            
    parameters = {'weight': w, 'bias': b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients,cost_list
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T, x_test)+b)
    y_predict = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_predict[0,i] = 0 
        else : 
            y_predict[0,i] = 1 
            
    return y_predict
        
def logistic_regression(x_train, y_train,x_test, y_test,learning_rate,num_of_iterarion):
    dimension = x_train.shape[0]
    w,b = init_param(dimension)
    
    parameters, gradients, cost_list = train(w,b,x_train,y_train, learning_rate,num_of_iterarion)
    
    y_prediction_test = predict(parameters['weight'], parameters['bias'], x_test)    
    y_prediction_train = predict(parameters['weight'], parameters['bias'], x_train)
    
     # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.001, num_of_iterarion = 200)
