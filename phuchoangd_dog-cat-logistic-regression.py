#plt.style.use('ggplot')

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
cat_cascade_extend = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalcatface_extended.xml')
input_train = '../input/dogs-vs-cats/train/train/'

input_test  = '../input/dogs-vs-cats/test/test/'



scale_factory = 1.05

neighbor = 5

img_size = 128

num_classes = 2

img_resize = (img_size, img_size)
img_path = input_train + 'cat.2.jpg'

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
x_data = np.asarray(train_images) #converting the list of images to numpy array.

y_data = np.asarray(train_labels)
X_train = x_data[:15000]

Y_train = y_data[:15000]
X_train = X_train/255. #(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) <=> x_train/255.
Y_train = Y_train.reshape(Y_train.shape[0], 1)
print(X_train.shape)

print(Y_train.shape)
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)
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
def init_weights_and_bias(dimension):

    weight = np.full((dimension, 1), 0.01)

    bias = 0.0

    return weight, bias
def sigmoid(z):

    y_head = 1/(1+np.exp(-z)) #1/(1+e^-x)

    return y_head
def forward_backward_propagation(weight, bias, x_train, y_train):

    # forward propagation

    z = np.dot(weight.T, x_train) + bias

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #-(y^)*log(ŷhat)-(1-y)*log(1-ŷhat)

    cost = (np.sum(loss))/x_train.shape[1] #x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost, gradients
def update_weight(weight, bias, x_train, y_train, learning_rate, number_of_iterarion):

    cost_list = []

    cost_list_plt = []

    index = []

    

    for i in range(number_of_iterarion):

        

        cost, gradients = forward_backward_propagation(weight, bias, x_train, y_train)

        cost_list.append(cost)

        

        weight = weight - learning_rate * gradients["derivative_weight"]

        bias = bias - learning_rate * gradients["derivative_bias"]

        

        if i > 0 and i % 500 == 0:

            cost_list_plt.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %.4f" %(i, cost))

                

    parameters = {"weight": weight, "bias": bias}

    

    plt.plot(index, cost_list_plt)

    plt.xticks(index, rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

        

    return parameters, gradients, cost_list
def predict(weight, bias, x_test):

    

    z = sigmoid(np.dot(weight.T, x_test) + bias)

    Y_prediction = np.zeros((1, x_test.shape[1]))



    for i in range(z.shape[1]):

        if z[0, i] <= 0.5:

            Y_prediction[0, i] = 0 #Cat

        else:

            Y_prediction[0, i] = 1 #Dog



    return Y_prediction
def logistic_regression(x_train, y_train, learning_rate, num_iterations):



    dimension =  x_train.shape[0]

    weight, bias = init_weights_and_bias(dimension)



    parameters, gradients, cost_list = update_weight(weight, bias, x_train, y_train, learning_rate, num_iterations)

    

    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)

    

    print("Test Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100, 2)))

    print("Train Accuracy: {} %".format(round(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100, 2)))

    

    return y_prediction_train
train_result = logistic_regression(x_train, y_train, learning_rate=0.0002, num_iterations=5000)
train_result2 = logistic_regression(x_train, y_train, learning_rate=0.0002, num_iterations=10000)