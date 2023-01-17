import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.image as implt

import os

import seaborn as sns

import cv2 as cv



from PIL import Image



import warnings

# filter warnings

warnings.filterwarnings('ignore')
train_path = "../input/genderdetectionface/dataset1/train"

test_path = "../input/genderdetectionface/dataset1/test"



train_woman = sorted(os.listdir(train_path +'/woman'))

train_man =  sorted(os.listdir(train_path +'/man'))



test_woman = sorted(os.listdir(test_path +'/woman'))

test_man =  sorted(os.listdir(test_path +'/man'))

category_names = sorted(os.listdir(train_path))

img_pr_cat = []

for category in category_names:

    folder = train_path + '/' + category

    img_pr_cat.append(len(os.listdir(folder)))

sns.barplot(x=category_names, y=img_pr_cat).set_title("Number of training images:")
category_names = sorted(os.listdir(test_path))

img_pr_cat = []

for category in category_names:

    folder = test_path + '/' + category

    img_pr_cat.append(len(os.listdir(folder)))

sns.barplot(x=category_names, y=img_pr_cat).set_title("Number of test images:")

img_1 = implt.imread(train_path +'/woman/face_389.jpg')

img_2 = implt.imread(train_path +'/man/face_486.jpg')



plt.subplot(1, 2, 1)

plt.title('woman')

plt.imshow(img_1)       

plt.subplot(1, 2, 2)

plt.title('man')

plt.imshow(img_2) 
img_size = 50

women_faces = []

men_faces = [] 

label = []



for i in train_woman:

        if os.path.isfile(train_path +'/woman/'+ i):

            faces = Image.open(train_path +'/woman/'+ i).convert('L') #converting grey scale            

            faces = faces.resize((img_size,img_size), Image.ANTIALIAS) #resizing to 50,50

            faces = np.asarray(faces)/255.0 #normalizing images

            women_faces.append(faces)  

            label.append(1) #label 1 for women

 

for i in train_man:

        if os.path.isfile(train_path+'/man/'+ i):

            faces = Image.open(train_path+'/man/'+ i).convert('L')

            faces = faces.resize((img_size,img_size), Image.ANTIALIAS)

            faces = np.asarray(faces)/255.0 #normalizing images

            men_faces.append(faces)  

            label.append(0) #label 0 for men          

           

x_train = np.concatenate((women_faces,men_faces),axis=0) # training dataset

x_train_label = np.asarray(label)# label array containing 0 and 1

x_train_label = x_train_label.reshape(x_train_label.shape[0],1)



print("women_faces:",np.shape(women_faces) , "men_faces:",np.shape(men_faces))

print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
img_size = 50

women_faces = []

men_faces = [] 

label = []



for i in test_woman:

        if os.path.isfile(test_path +'/woman/'+ i):

            faces = Image.open(test_path +'/woman/'+ i).convert('L')            

            faces = faces.resize((img_size,img_size), Image.ANTIALIAS)

            faces = np.asarray(faces)/255.0

            women_faces.append(faces)  

            label.append(1)     

 

for i in test_man:

        if os.path.isfile(test_path+'/man/'+ i):

            faces = Image.open(test_path+'/man/'+ i).convert('L')

            faces = faces.resize((img_size,img_size), Image.ANTIALIAS)

            faces = np.asarray(faces)/255.0            

            men_faces.append(faces)

            label.append(0)                       



x_test = np.concatenate((women_faces,men_faces),axis=0) # test dataset

x_test_label = np.asarray(label) # corresponding labels

x_test_label = x_test_label.reshape(x_test_label.shape[0],1)



print("women_faces:",np.shape(women_faces), "men_faces:",np.shape(men_faces))

print("test_dataset:",np.shape(x_test), "test_values:",np.shape(x_test_label))
x = np.concatenate((x_train,x_test),axis=0) #train_data

y = np.concatenate((x_train_label,x_test_label),axis=0) #test data

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) #flatten 3D image array to 2D

print("images:",np.shape(x), "labels:",np.shape(y))
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

number_of_train = X_train.shape[0]

number_of_test = X_test.shape[0]



print("train number:",number_of_train, "test number:",number_of_test)
x_train = X_train.T

x_test = X_test.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b



def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head



def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b    

    y_head = sigmoid(z)    

    loss = -(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head)        

    cost = (np.sum(loss))/x_train.shape[1]  # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                   # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients



def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 50 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is woman (y_head=1),

    # if z is smaller than 0.5, our prediction is man (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # 2500

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    train_acc_lr = round((100 - np.mean(np.abs(y_prediction_train - y_train)) * 100),2)

    test_acc_lr = round((100 - np.mean(np.abs(y_prediction_test - y_test)) * 100),2)

    # Print train/test Errors

    print("train accuracy: %", train_acc_lr)

    print("test accuracy: %", test_acc_lr)

    return train_acc_lr, test_acc_lr

    



train_acc_lr, test_acc_lr = logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 500)

#you can adjust learning_rate and num_iteration to check how the result is affected

#(for learning rate, try exponentially lower values:0.001 etc.) 

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



logreg = LogisticRegression()

test_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_logregsk = round(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

test_acc_perceptron = round(perceptron.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)* 100, 2)

train_acc_perceptron = round(perceptron.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)* 100, 2)
models = pd.DataFrame({

    'Model': ['LR without sklearn','LR with sklearn', 'Perceptron'],

    'Train Score': [train_acc_lr, train_acc_logregsk, train_acc_perceptron],

    'Test Score': [test_acc_lr, test_acc_logregsk, test_acc_perceptron]})

models.sort_values(by='Test Score', ascending=False)