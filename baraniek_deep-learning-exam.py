# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import matplotlib.image as implt
from PIL import Image 
import seaborn as sns
import cv2 as cs2

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path = "/kaggle/input/alien-vs-predator-images/data/train"
test_path = "/kaggle/input/alien-vs-predator-images/data/validation"
alien_test = "/kaggle/input/alien-vs-predator-images/data/validation/alien"
alien_train = "/kaggle/input/alien-vs-predator-images/data/train/alien"
predator_train = "/kaggle/input/alien-vs-predator-images/data/train/predator"
predator_test = "/kaggle/input/alien-vs-predator-images/data/validation/predator"
category_names = os.listdir(train_path) # output: ['humans', 'horses']
nb_categories = len(category_names) # output: 2
train_images = []

for category in category_names:
    folder = train_path + "/" + category
    train_images.append(len(os.listdir(folder)))

sns.barplot(y=category_names, x=train_images).set_title("Number Of Training Images Per Category");
test_images = []
for category in category_names:
    folder = test_path + "/" + category
    test_images.append(len(os.listdir(folder)))

sns.barplot(y=category_names, x=train_images).set_title("Number Of Testing Images Per Category");
img1 = implt.imread(predator_train + "/0.jpg")
img2 = implt.imread(alien_train +  "/0.jpg")

plt.subplot(1, 2, 1)
plt.title('predator')
plt.axis("off")
plt.imshow(img1)       
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title('alien')
plt.imshow(img2)
plt.show()
img_size = 50
train_predator = []
train_alien = []
label = []

for i in os.listdir(predator_train): # all train human images
    if os.path.isfile(train_path + "/predator/" + i): # check image in file
        predator = Image.open(train_path + "/predator/" + i).convert("L") # converting grey scale 
        predator = predator.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        predator = np.asarray(predator)/255 # bit format
        train_predator.append(predator)
        label.append(1)
        
for i in os.listdir(alien_train): # all train horse images
    if os.path.isfile(train_path + "/alien/" + i): # check image in file
        alien = Image.open(train_path + "/alien/" + i).convert("L") # converting grey scale 
        alien = alien.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        alien = np.asarray(alien)/255 # bit format
        train_alien.append(alien)
        label.append(0)
x_train = np.concatenate((train_predator,train_alien),axis=0) # training dataset
x_train_label = np.asarray(label) # label array containing 0 and 1
x_train_label = x_train_label.reshape(x_train_label.shape[0],1)

print("predator:",np.shape(train_predator) , "alien:",np.shape(train_alien))
print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
np.shape(x_train_label)
img_size = 50
test_predator = []
test_alien = []
label = []

for i in os.listdir(predator_test): # all train human images
    if os.path.isfile(test_path + "/predator/" + i): # check image in file
        predator = Image.open(test_path + "/predator/" + i).convert("L") # converting grey scale 
        predator = predator.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        predator = np.asarray(predator)/255 # bit format
        test_predator.append(predator)
        label.append(1)
        
for i in os.listdir(alien_test): # all train horse images
    if os.path.isfile(test_path + "/alien/" + i): # check image in file
        alien = Image.open(test_path + "/alien/" + i).convert("L") # converting grey scale 
        alien = alien.resize((img_size,img_size), Image.ANTIALIAS) # resizing to 50,50
        alien = np.asarray(alien)/255 # bit format
        test_alien.append(alien)
        label.append(0)
x_test = np.concatenate((test_predator,test_alien),axis=0) # training dataset
x_test_label = np.asarray(label) # label array containing 0 and 1
x_test_label = x_test_label.reshape(x_test_label.shape[0],1)

print("humans:",np.shape(test_predator) , "horses:",np.shape(test_alien))
print("test_dataset:",np.shape(x_test), "test_values:",np.shape(x_test_label))
x = np.concatenate((x_train,x_test),axis=0) # count: 1027+256= 1283  | train_data
# x.shape: 
#   output = (894,2500)
y = np.concatenate((x_train_label,x_test_label)) # count: 1027+256= 1283 | test_data
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) # flatten 3D image array to 2D, count: 50*50 = 2500
print("images:",np.shape(x), "labels:",np.shape(y))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state = 42)
number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
X = np.concatenate((x[204:], x[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)
#LOGISTIC REGRESSION

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
def forward_propagation(w,b,x_train,y_train):
    x = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    return cost
def backward_propagation(w,b,x_train,y_train):
    türev_weight = (np.dot(x_train,((y_head-y_train).T))/x_train.shape[1])
    türev_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradyan = {"türev_weinght":türev_weight,"türev_bias":türev_bias}
    return gradyan
def update(w,b,x_train,y_train,learning_rate,iter_no):
    cost_list1 = []
    cost_list2 = []
    index = []
    for i in range(iter_no):
        cost,gradyan = forward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w-learning_rate*gradyan["türev_weight"]
        b = b-learning_rate*gradyan["türev_bias"]
        if i %50 == 0 : 
            cost_list2.append(cost)
            index.append(i)
            print("cost after iter %i: %f"%(i,cost))
    parametre = {"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation = "vertical")
    plt.xlabel("iter_no")
    plt.ylabel("cost")
    plt.show()
    return parametre,gradyan,cost_list
def prediction(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_predict = np.zeros((1,x_test.shape[1]))
    for i in range(x.shape[1]):
        if z[0,i]<=0.5:
            y_predict[0,l] = 0
        else:
            y_predict[0,i] = 1
    return y_predict
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, iter_no):
    # initialize
    dimension = x_train.shape[0] # 2500
    w,b = initialize_weights_and_bias(dimension)
    parametre, gradyan, cost_list = update(w,b,x_train,y_train,learning_rate,iter_no)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    test_acc_lr = round((100 - np.mean(np.abs(y_prediction_test - y_test)) * 100),2)
    train_acc_lr = round((100 - np.mean(np.abs(y_prediction_train - y_train))*100),2)
    
    # Print train/test Errors
    print("train accuracy: %", train_acc_lr)
    print("test accuracy: %", test_acc_lr)
    return train_acc_lr, test_acc_lr
logistic(x_train,y_train,x_test,y_test,learning_rate=0.01,iter_no = 1500)
from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter = 150)
logreg(x_train,y_train,x_test,y_test)
