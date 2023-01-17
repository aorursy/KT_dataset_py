%config IPCompleter.greedy=True

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from PIL import Image

import cv2 

from random import randint

from sklearn.utils import shuffle

from keras.utils import np_utils

from tqdm import tqdm #Progress Bars
input_train = '../input/dogs-vs-cats/train/train/'

input_test  = '../input/dogs-vs-cats/test/test/'



scale_factory = 1.05

neighbor = 5

img_size = 128

num_classes = 2

img_resize = (img_size, img_size)
cat_cascade_extend = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalcatface_extended.xml')
img_path = input_train + 'cat.10.jpg'

# Show test data

def show_image(img_path):

    img = cv2.imread(img_path)

    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(cv_rgb)

    plt.show()

    

show_image(img_path)
def resize_to_square(image, size):

    h, w = image.shape

    ratio = size / max(h, w)

    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)

    return resized_image
def padding(image, min_height, min_width):

    h, w = image.shape



    if h < min_height:

        h_pad_top = int((min_height - h) / 2.0)

        h_pad_bottom = min_height - h - h_pad_top

    else:

        h_pad_top = 0

        h_pad_bottom = 0



    if w < min_width:

        w_pad_left = int((min_width - w) / 2.0)

        w_pad_right = min_width - w - w_pad_left

    else:

        w_pad_left = 0

        w_pad_right = 0

        

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))
def get_images(train_path): 

    Labels = []  # 0 for Building , 1 dog, 2 for cat

    Images = []

    Filenames = []



    for file in tqdm(os.listdir(input_train)): #Main Directory where each class label is present as folder name.            

        words = file.split(".") 

        

        if words[0] == 'cat':

            drop = False

            image = cv2.imread(input_train + file, cv2.IMREAD_GRAYSCALE)

            gray = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

            cat_image_extends = cat_cascade_extend.detectMultiScale(image, scale_factory, neighbor)

            

            for (x,y,w,h) in cat_image_extends:   

                image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)

                roi_gray = gray[y:y+h, x:x+w]

                roi_color = image[y:y+h, x:x+w]

                image_resized = cv2.resize(roi_color, (img_size, img_size)) 



                Images.append(image_resized)

                Labels.append(1.0)

                Filenames.append(file)

                drop = True

                

            if drop == False:

                image = cv2.imread(input_train + file, cv2.IMREAD_GRAYSCALE)

                image = resize_to_square(image, img_size)

                image_resized = padding(image, img_size, img_size)

                

                Images.append(image_resized)

                Labels.append(1.0)

                Filenames.append(file)   

                

        elif words[0] == 'dog':

            image = cv2.imread(input_train + file, cv2.IMREAD_GRAYSCALE)#COLOR_BGR2RGB, IMREAD_UNCHANGED, IMREAD_GRAYSCALE

            image = resize_to_square(image, img_size)

            image = padding(image, img_size, img_size)

            

            Images.append(image)

            Labels.append(0.0)

            Filenames.append(file) 

    

    return shuffle(Images, Labels, Filenames, random_state=817328462) #Shuffle the dataset you just prepared.
train_images, train_labels, train_filenames = get_images(input_train) #Extract the training images from the folders.
def get_classlabel(class_code):

    labels = {1:'cats', 0:'dogs'}

    

    return labels[class_code]



f, ax = plt.subplots(3,3) 

f.subplots_adjust(0,0,3,3)

for i in range(0,3,1):

    for j in range(0,3,1):

        rnd_number = randint(0, len(train_images))

        ax[i,j].imshow(train_images[rnd_number])

        ax[i,j].set_title(get_classlabel(train_labels[rnd_number]))

        ax[i,j].axis('off')
x_data = np.asarray(train_images) #converting the list of images to numpy array.

y_data = np.asarray(train_labels)
X_train = x_data[:20000]

Y_train = y_data[:20000]
X_train = X_train/255. #(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) <=> x_train/255.
Y_train = Y_train.reshape(Y_train.shape[0], 1)
print(X_train.shape)

print(Y_train.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)
x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))

x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
number_of_train = x_train.shape[0]

number_of_test = x_test.shape[0]



x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])

x_test_flatten = x_test .reshape(number_of_test,x_test.shape[1]*x_test.shape[2])



print("X train flatten", x_train_flatten.shape)

print("X test flatten", x_test_flatten.shape)
x_train = x_train_flatten.T

x_test = x_test_flatten.T

y_test = y_test.T

y_train = y_train.T



print("x train: ", x_train.shape)

print("x test: ", x_test.shape)

print("y train: ", y_train.shape)

print("y test: ", y_test.shape)
# calculation of z

#z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
# intialize parameters and layer sizes

def initialize_parameters_and_layer_sizes(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.01,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.01,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters
def forward_propagation(x_train, parameters):



    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]

    A1 = np.tanh(Z1)

    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

    

    return A2, cache
# Compute cost

def compute_cost(A2, Y, parameters):

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost
# Backward Propagation

def backward_propagation(parameters, cache, X, Y):



    dZ2 = cache["A2"]-Y

    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]

    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))

    dW1 = np.dot(dZ1,X.T)/X.shape[1]

    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]

    grads = {"dweight1": dW1,

             "dbias1": db1,

             "dweight2": dW2,

             "dbias2": db2}

    return grads
# update parameters

def update_parameters_neural_network(parameters, grads, learning_rate = 0.002):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters
# prediction

def predict_neural_network(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation(x_test,parameters)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(A2.shape[1]):

        if A2[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
# 2 - Layer neural network

def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):

    cost_list = []

    index_list = []

    #initialize parameters and layer sizes

    parameters = initialize_parameters_and_layer_sizes(x_train, y_train)



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation(x_train,parameters)

        # compute cost

        cost = compute_cost(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation(parameters, cache, x_train, y_train)

         # update parameters

        parameters = update_parameters_neural_network(parameters, grads)

        

        if i % 500 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

        if cost <= 0.09:

            break

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # predict

    y_prediction_test = predict_neural_network(parameters,x_test)

    y_prediction_train = predict_neural_network(parameters,x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters
parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=5000)
parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=10000)