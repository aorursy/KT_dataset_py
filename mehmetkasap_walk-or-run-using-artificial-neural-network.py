# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# First, look at everything.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import cv2
import os
import random
import matplotlib.pylab as plt
from glob import glob
import pandas as pd
import numpy as np
# in order to import an image
from IPython.display import Image
im1 = Image("../input/walk_or_run_train/train/run/run_00061c18.png")
im1
# TRAIN

# ../input/
PATH = os.path.abspath(os.path.join('..', 'input'))

# TRAIN_RUN

# ../input/walk_or_run_train/train/run
train_run_images = os.path.join(PATH, "walk_or_run_train", "train", 'run')
# ../input/walk_or_run_train/train/run/*.png
train_run = glob(os.path.join(train_run_images, "*.png"))

# TRAIN_WALK

# ../input/walk_or_run_train/train/walk
train_walk_images = os.path.join(PATH, "walk_or_run_train", "train", 'walk')
# ../input/walk_or_run_train/train/walk/*.png
train_walk = glob(os.path.join(train_walk_images, "*.png"))

# ADD TRAIN_WALK AND TRAIN_RUN INTO A DATAFRAME

train = pd.DataFrame()
train['file'] = train_run + train_walk
train.head()
# TEST

# ../input/
PATH = os.path.abspath(os.path.join('..', 'input'))

# TEST_RUN

# ../input/walk_or_run_test/test/run
test_run_images = os.path.join(PATH, "walk_or_run_test", "test", 'run')
# ../input/walk_or_run_test/test/run/*.png
test_run = glob(os.path.join(test_run_images, "*.png"))

# TEST_WALK

# ../input/walk_or_run_test/test/walk
test_walk_images = os.path.join(PATH, "walk_or_run_test", "test", 'walk')
# ../input/walk_or_run_test/test/walk/*.png
test_walk = glob(os.path.join(test_walk_images, "*.png"))

test = pd.DataFrame()
test['file'] = test_run + test_walk
test.shape
#TRAIN LABELS

train['label'] = [1 if i in train_run else 0 for i in train['file']]
train.head()
#TEST LABELS

test['label'] = [1 if i in test_run else 0 for i in test['file']]
test.tail()
# TRAIN RUN AND WALK IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(train_run[2]))

plt.subplot(122)
plt.imshow(cv2.imread(train_walk[0]))
# TEST RUN AND WALK IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(test_run[12]))

plt.subplot(122)
plt.imshow(cv2.imread(test_walk[5]))
train.shape # features: file and layer
# Each image shape is (224,224,3) (lets consider here only the first image)
cv2.imread(train.file[0]).shape
# convert the dimension of an image into (224x224,3) namely: (50176,3)
image1 = cv2.imread(train.file[0]).reshape(224*224,3)
image1.shape

# here 3 represent three dimension of a color pixel
# for example RGB(23,24,122)
# convert this 3 dimensioanal color into an integer color value = R + G*(256) + B*(256^2)
# for this example, lets use first pixel RGB values
r1 = image1[0][0] # Red value of the first pixel of the first image
g1 = image1[0][1] # Green value of the first pixel of the first image
b1 = image1[0][2] # Blue value of the first pixel of the first image

# now convert this 3 dimensional color value into an integer color value (of the first pixel)
first_pixel_integer_color_value = r1+(256*g1)+(256*256*b1)
first_pixel_integer_color_value
# create x_train3D and reshape it (coverting images into array)

x_train3D = []
for i in range(0,600):
    x_train3D.append(cv2.imread(train.file[i]).reshape(224*224,3))
    
x_train3D = np.asarray(x_train3D) # to make it array
x_train3D = x_train3D/1000 # for scaling

# create y_train
y_train = train.label
y_train = np.asarray(y_train) # to make it array
x_train3D.shape
# create x_train 
# integer color value = R + G*(256) + B*(256^2)
x_train = np.zeros((600,50176))
for i in range(0,600):
    for j in range(0,50176):
        x_train[i,j] = ((x_train3D[i][j][0]+(256*x_train3D[i][j][1])+(256*256*x_train3D[i][j][2]))/10000000)

x_train = np.asarray(x_train) # to make it array
x_train.shape
y_train.shape
# ORIJINAL IMAGES
# TRAIN RUN IMAGES
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(cv2.imread(train_run[0]))

plt.subplot(122)
plt.imshow(cv2.imread(train_run[1]))
# MODIFIED ONES

img_size = 224
plt.figure(figsize=(16,16))
plt.subplot(1, 2, 1)
plt.imshow(x_train[0].reshape(img_size, img_size))
plt.subplot(1, 2, 2)
plt.imshow(x_train[1].reshape(img_size, img_size))
test.shape
# create x_test and reshape it (coverting images into array)
x_test3D = []
for i in range(0,141):
    x_test3D.append(cv2.imread(test.file[i]).reshape(224*224,3))

x_test3D = np.asarray(x_test3D) # to make it array

# create y_test
y_test = test.label
y_test = np.asarray(y_test) # to make it array
x_test3D.shape
# create x_test 
# integer color value = R + G*(256) + B*(256^2)
x_test = np.zeros((141,50176))
for i in range(0,141):
    for j in range(0,50176):
        x_test[i,j] = ((x_test3D[i][j][0]+(256*x_test3D[i][j][1])+(256*256*x_test3D[i][j][2]))/10000000)

x_test = np.asarray(x_test) # to make it array
x_test.shape
y_test.shape
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=42, max_iter = 150, C = 0.1)
LR.fit(x_train,y_train)
print("test accuracy: {} ".format(LR.score(x_test, y_test)))
print("train accuracy: {} ".format(LR.score(x_train, y_train)))
from sklearn.linear_model import SGDClassifier
SGDC = SGDClassifier(max_iter = 100)
SGDC.fit(x_train, y_train)
print("test accuracy: {} ".format(SGDC.score(x_test, y_test)))
print("train accuracy: {} ".format(SGDC.score(x_train, y_train)))
x_train = x_train.T # it means we have 600 images with 50176 pixels each
x_train.shape
y_train = y_train.T
y_train.shape
x_test = x_test.T
y_test = y_test.T
x_test.shape
def initialize_weights_bias(x_train, nodes=3):
    w1 = np.random.rand(nodes,x_train.shape[0])*0.1
    b1 = np.zeros((nodes,1))
    w2 = np.random.rand(1,nodes)*0.1
    b2 = np.zeros((1,1))
    return w1, b1, w2, b2
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y
def forward_propogation(x_train,w1,b1,w2,b2):
    Z1 = np.dot(w1,x_train)+b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(w2,A1)+b2
    A2 = sigmoid(Z2)
    
    return Z1,A1,Z2,A2
def calculate_cost(A2, Y):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost
# Backward Propagation
def backward_propagation(w1, b1, w2, b2, A1, A2, X, Y):

    dZ2 = A2-Y
    dW2 = np.dot(dZ2,A1.T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(w2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    
    return dW1, db1, dW2, db2 
def update_parameters(w1,b1,w2,b2,dW1,db1,dW2,db2,learning_rate=0.01):
    w1 = w1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    w2 = w2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    return w1,b1,w2,b2    
def two_layer_Neural_Netwok(x_train, x_test, y_train, y_test, max_iter = 150):
    cost_list = []
    index_list = []
    
    w1,b1,w2,b2 = initialize_weights_bias(x_train, nodes=3)
    for i in range(0,max_iter):
        Z1,A1,Z2,A2 = forward_propogation(x_train,w1,b1,w2,b2)
        # cost = calculate_cost(A2, y_train.T)
        dW1, db1, dW2, db2 = backward_propagation(w1, b1, w2, b2, A1, A2, x_train, y_train)
        w1,b1,w2,b2 = update_parameters(w1,b1,w2,b2,dW1,db1,dW2,db2,learning_rate=0.01)
    
        #if i % 10 == 0:
        #    cost_list.append(cost)
        #    index_list.append(i)
        #    print ("Cost after iteration {0}: {1}".format(i, cost))
    #plt.plot(index_list,cost_list)
    #plt.xticks(index_list,rotation='vertical')
    #plt.xlabel("Number of Iterarion")
    #plt.ylabel("Cost")
    #plt.show()
    
    # Now, lets make prediciton using updated parameters
    Z1,A1,Z2,A2 = forward_propogation(x_test,w1,b1,w2,b2)
    y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(x_test.shape[1]):
        if A2[0,i]<=0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction - y_test)) * 100))
    return w1,b1
two_layer_Neural_Netwok(x_train, x_test, y_train, y_test, max_iter = 150)
# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN
# 2 hidden layers we have (totally 3 layers, together with the output)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 200)
# accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
classifier.fit(x_train, y_train, epochs = 200)
print('test score of 3-Layer ANN: ', classifier.score(x_test,y_test)) 
print('train score of 3-Layer ANN: ', classifier.score(x_train,y_train)) 
# Evaluating the ANN
# 3 hidden layer
# each has 64, 30, 3 neurons (nodes) respectively
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 6, kernel_initializer = 'random_uniform', activation = 'tanh', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'random_uniform', activation = 'tanh'))
    classifier.add(Dense(units = 3, kernel_initializer = 'random_uniform', activation = 'tanh'))
    classifier.add(Dense(units = 1, kernel_initializer = 'random_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 300)
# accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
classifier.fit(x_train, y_train, epochs = 300)
print('test score of 4-Layer ANN: ', classifier.score(x_test,y_test)) 
print('train score of 4-Layer ANN: ', classifier.score(x_train,y_train)) 