import os

import tqdm

import matplotlib.pyplot as plt

import cv2

import numpy as np



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

img_size=150

training_data_cat=[]

train_path_cats = '../input/cat-and-dog/training_set/training_set/cats'

for path in os.listdir(train_path_cats):

    if '.jpg' in path:

        img_array=cv2.imread(os.path.join(train_path_cats,path),cv2.IMREAD_GRAYSCALE)

        new_array= cv2.resize(img_array,(img_size,img_size))

        training_data_cat.append(new_array)



training_data_dog=[]

train_path_dogs = '../input/cat-and-dog/training_set/training_set/dogs'

for path  in os.listdir(train_path_dogs):

    if '.jpg' in path:

        img_array=cv2.imread(os.path.join(train_path_dogs,path),cv2.IMREAD_GRAYSCALE)

        new_array= cv2.resize(img_array,(img_size,img_size))

        training_data_dog.append(new_array)

        

test_data_cat=[]

test_path_cats = '../input/cat-and-dog/test_set/test_set/cats'

for path in os.listdir(test_path_cats):

    if '.jpg' in path:

        img_array=cv2.imread(os.path.join(test_path_cats,path),cv2.IMREAD_GRAYSCALE)

        new_array= cv2.resize(img_array,(img_size,img_size))

        test_data_cat.append(new_array)



test_data_dog=[]

test_path_dogs = '../input/cat-and-dog/test_set/test_set/dogs'

for path  in os.listdir(test_path_dogs):

    if '.jpg' in path:

        img_array=cv2.imread(os.path.join(test_path_dogs,path),cv2.IMREAD_GRAYSCALE)

        new_array= cv2.resize(img_array,(img_size,img_size))

        test_data_dog.append(new_array)

training_data_x=np.concatenate((training_data_dog,training_data_cat),axis=0)

training_data_y=np.concatenate((np.ones(len(training_data_dog)),np.zeros(len(training_data_cat))),axis=0)

test_data_x=np.concatenate((test_data_dog,test_data_cat),axis=0)

test_data_y=np.concatenate((np.ones(len(test_data_dog)),np.zeros(len(test_data_cat))),axis=0)
training_data_x_flatten = training_data_x.reshape(training_data_x.shape[0],training_data_x.shape[1]*training_data_x.shape[2])

test_data_x_flatten = test_data_x.reshape(test_data_x.shape[0],test_data_x.shape[1]*test_data_x.shape[2])

training_data_X=training_data_x_flatten.T

training_data_Y=training_data_y.T.reshape(8005,1)

test_data_X=test_data_x_flatten.T

test_data_Y=test_data_y.T.reshape(2023,1)
print("x train: ",training_data_X.shape)

print("x test: ",test_data_X.shape)

print("y train: ",training_data_Y.shape)

print("y test: ",test_data_Y.shape)
def initilaze_weights_and_bias(dimension):

    w= np.full((dimension,1),0.01)          #full function'ı dimesion x 1 lik bir matris oluşturup içine 0.01 değerini dolduruyor.

    b=0

    return w,b



def sigmoid(z):

    y_head= 1/(1+np.exp(-z))

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    z=np.dot(w.T,x_train)+b   #weight ve px1 i çarpıp bias ile topladıktan sonra z değeri elde ediyoruz.

    y_head=sigmoid(z)

    y_head_1 = 1-y_head

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(y_head_1)

    print(y_train)

    print(1-y_train)

    print(y_head[0][0])

    print(1-y_head[0][0])

    print(np.log(y_head))

    print(np.log(y_head_1))

    print(loss)

    cost= (np.sum(loss))/x_train.shape[1]

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients



w,b = initilaze_weights_and_bias(22500)



cost,gradients = forward_backward_propagation(w,b,training_data_X,training_data_Y)



print(cost)



def forward_backward_propagation(w,b,x_train,y_train):

    z=np.dot(w.T,x_train)+b   #weight ve px1 i çarpıp bias ile topladıktan sonra z değeri elde ediyoruz.

    y_head=sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost= (np.sum(loss))/x_train.shape[1]

    

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias":derivative_bias}

    return cost,gradients



def update(w,b,x_train,y_train,learning_rate,number_of_iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    for i in range(number_of_iteration):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 ==0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i, cost))

    parameters = {"weight":w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number Of Itaration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list  



def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

            

    return Y_prediction   



def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    dimension = x_train.shape[0]  #4096

    w, b = initilaze_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(training_data_X,training_data_Y,test_data_X,test_data_Y,learning_rate=0.01,num_iterations=20)