import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.image as implt

import seaborn as sns



from PIL import Image

import cv2



import warnings

warnings.filterwarnings('ignore')



import os

cwd = os.getcwd()

os.chdir(cwd)

print(os.listdir("../input"))

# Load at the data

data_test ='../input/cat-and-dog/test_set/test_set/'

data_train ='../input/cat-and-dog/training_set/training_set'



train_dogs = sorted(os.listdir(data_train +'/dogs'))

train_cats =  sorted(os.listdir(data_train +'/cats'))



test_dogs = sorted(os.listdir(data_test +'/dogs'))

test_cats =  sorted(os.listdir(data_test +'/cats'))
img_1 = implt.imread(data_train +'/dogs/dog.3851.jpg')

img_2 = implt.imread(data_train +'/cats/cat.3041.jpg')



plt.subplot(1, 2, 1)

plt.title('Dog')

plt.imshow(img_1)       

plt.subplot(1, 2, 2)

plt.title('Cat')

plt.imshow(img_2) 
img_size = 64

dogs_images = []

cats_images = [] 

label = []



for i in train_dogs:

    if '.jpg' in i:

        if os.path.isfile(data_train +'/dogs/'+ i):

            images = Image.open(data_train +'/dogs/'+ i).convert('L') #converting grayscale         

            images = images.resize((img_size,img_size), Image.ANTIALIAS) #resizing to 64,64

            images = np.asarray(images)/255.0 #normalizing images

            dogs_images.append(images)  

            label.append(1) #label 1 for dogs

                    

for i in train_cats:

    if '.jpg' in i:

        if os.path.isfile(data_train+'/cats/'+ i):

            images = Image.open(data_train+'/cats/'+ i).convert('L')

            images = images.resize((img_size,img_size), Image.ANTIALIAS)

            images = np.asarray(images)/255.0 #normalizing images

            cats_images.append(images)  

            label.append(0) #label 0 for cats          

           

x_train = np.concatenate((dogs_images,cats_images),axis=0) # training dataset

x_train_label = np.asarray(label)# label array containing 0 and 1

x_train_label = x_train_label.reshape(x_train_label.shape[0],1)



print("dogs_images:",np.shape(dogs_images) , "cats_images:",np.shape(cats_images))

print("train_dataset:",np.shape(x_train), "train_values:",np.shape(x_train_label))
img_size = 64

dogs_images = []

cats_images = [] 

label = []



for i in test_dogs:

    if '.jpg' in i:

        if os.path.isfile(data_test +'/dogs/'+ i):

            images = Image.open(data_test +'/dogs/'+ i).convert('L') #converting grayscale            

            images = images.resize((img_size,img_size), Image.ANTIALIAS) #resizing to 64,64

            images = np.asarray(images)/255.0 #normalizing images

            dogs_images.append(images)  

            label.append(1) #label 1 for dogs

 

for i in test_cats:

    if '.jpg' in i:

        if os.path.isfile(data_test +'/cats/'+ i):

            images = Image.open(data_test +'/cats/'+ i).convert('L')

            images = images.resize((img_size,img_size), Image.ANTIALIAS)

            images = np.asarray(images)/255.0 #normalizing images

            cats_images.append(images)  

            label.append(0) #label 0 for cats       

            

x_test = np.concatenate((dogs_images,cats_images),axis=0) # test dataset

x_test_label = np.asarray(label) # corresponding labels

x_test_label = x_test_label.reshape(x_test_label.shape[0],1)



print("dogs_images:",np.shape(dogs_images), "cats_images:",np.shape(cats_images))

print("test_dataset:",np.shape(x_test), "test_values:",np.shape(x_test_label))
x = np.concatenate((x_train,x_test),axis=0) #train_data

y = np.concatenate((x_train_label,x_test_label),axis=0) #test data

x = x.reshape(x.shape[0],x.shape[1]*x.shape[2]) #flatten 3D image array to 2D

print("images:",x.shape, "labels:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)



x_train = X_train.T

x_test = X_test.T

y_train = Y_train.T

y_test = Y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
def initialize_weights_and_bias(dimension):

    w=np.full((dimension,1),0.01)   

    b=0.0

    return w,b

def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head

def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)

    loss= -y_train*np.log(y_head)- (1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1]   #x_train.shape[1] is for scaling

    

    #backward propagation

    #x_train.shape[1] is for scaling

    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 

    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]

    gradients={'derivative_weight':derivative_weight,'derivative_bias':derivative_bias}

    

    return cost,gradients

def update(w,b,x_train,y_train,learnig_rate,number_of_iteration):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    #updating(learning) parametres is number_of_iteration times

    for i in range(number_of_iteration):

        #make fordward and backward propagation and find cost and gradients

        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        #update

        w=w-learnig_rate*gradients['derivative_weight']

        b=b-learnig_rate*gradients['derivative_bias']

        

        if i%50 == 0:

            cost_list2.append(cost)

            index.append(i)

            print('cost after iteration %i: %f:' %(i,cost))

    

    #we update(learn) parametres weights and bias

    parameters={'weight':w,'bias':b}

    plt.figure(figsize=(4,4))

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel('Number of iteration')

    plt.ylabel('Cost')

    plt.show()

    

    return parameters,gradients,cost_list

def predict(w,b,x_test):

    z=sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction=np.zeros((1,x_test.shape[1]))

    #We're making an estimate based on our condition.

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # 4096

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    train_acc_lr = round((100 - np.mean(np.abs(y_prediction_train - y_train)) * 100),2)

    test_acc_lr = round((100 - np.mean(np.abs(y_prediction_test - y_test)) * 100),2)



    from sklearn.metrics import confusion_matrix

    cm_test = confusion_matrix(y_test.T, y_prediction_test.T)

    cm_train = confusion_matrix(y_train.T, y_prediction_train.T)

    

    fig = plt.figure(figsize=(15,15))

    ax1 = fig.add_subplot(3, 3, 1) # row, column, position

    ax1.set_title('Confusion Matrix for Train Data')



    ax2 = fig.add_subplot(3, 3, 2) # row, column, position

    ax2.set_title('Confusion Matrix for Test Data')

    

    sns.heatmap(data=cm_train,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='BuPu')

    sns.heatmap(data=cm_test,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='BuPu')  

    

    

    # Print train/test Errors

    print("train accuracy: %", train_acc_lr)

    print("test accuracy: %", test_acc_lr)

    return train_acc_lr, test_acc_lr

    



train_acc_lr, test_acc_lr = logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.003, num_iterations = 500)
#Now we use sklearn libray 

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(random_state = 0)

lr.fit(x_train.T,y_train.T)



y_pred_test=lr.predict(x_test.T)

y_pred_train=lr.predict(x_train.T)



from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_test.T, y_pred_test)

cm_train = confusion_matrix(y_train.T, y_pred_train)



fig = plt.figure(figsize=(15,15))

ax1 = fig.add_subplot(3, 3, 1) # row, column, position

ax1.set_title('Confusion Matrix for Train Data')



ax2 = fig.add_subplot(3, 3, 2) # row, column, position

ax2.set_title('Confusion Matrix for Test Data')



sns.heatmap(data=cm_train,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='BuPu')

sns.heatmap(data=cm_test,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='BuPu')  



print('train accuracy: {}'.format(lr.score(x_train.T,y_train.T)))

print('test accuracy: {}'.format(lr.score(x_test.T,y_test.T)))
# intialize parameters and layer sizes

def initialize_parameters_and_layer_sizes_NN(x_train, y_train):

    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,

                  "bias1": np.zeros((3,1)),

                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,

                  "bias2": np.zeros((y_train.shape[0],1))}

    return parameters



def forward_propagation_NN(x_train, parameters):



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

def compute_cost_NN(A2, Y, parameters):

    logprobs = np.multiply(np.log(A2),Y)

    cost = -np.sum(logprobs)/Y.shape[1]

    return cost



# Backward Propagation

def backward_propagation_NN(parameters, cache, X, Y):



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

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],

                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],

                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],

                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}

    

    return parameters



# prediction

def predict_NN(parameters,x_test):

    # x_test is a input for forward propagation

    A2, cache = forward_propagation_NN(x_test,parameters)

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

    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)



    for i in range(0, num_iterations):

         # forward propagation

        A2, cache = forward_propagation_NN(x_train,parameters)

        # compute cost

        cost = compute_cost_NN(A2, y_train, parameters)

         # backward propagation

        grads = backward_propagation_NN(parameters, cache, x_train, y_train)

         # update parameters

        parameters = update_parameters_NN(parameters, grads)

        

        if i % 100 == 0:

            cost_list.append(cost)

            index_list.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    plt.plot(index_list,cost_list)

    plt.xticks(index_list,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    

    # predict

    y_prediction_test = predict_NN(parameters,x_test)

    y_prediction_train = predict_NN(parameters,x_train)

    

    from sklearn.metrics import confusion_matrix

    cm_test = confusion_matrix(y_test.T, y_prediction_test.T)

    cm_train = confusion_matrix(y_train.T, y_prediction_train.T)

    

    fig = plt.figure(figsize=(15,15))

    ax1 = fig.add_subplot(3, 3, 1) # row, column, position

    ax1.set_title('Confusion Matrix for Train Data')



    ax2 = fig.add_subplot(3, 3, 2) # row, column, position

    ax2.set_title('Confusion Matrix for Test Data')

    

    sns.heatmap(data=cm_train,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax1, cmap='BuPu')

    sns.heatmap(data=cm_test,annot=True, linewidth=0.7, linecolor='cyan', fmt='.0f', ax=ax2, cmap='BuPu')  



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    return parameters



parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=500)
# reshaping

x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T
# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import cross_val_score

from keras.models import Sequential # initialize neural network library

from keras.layers import Dense # build our layers library

def build_classifier():

    classifier = Sequential() # initialize neural network

    classifier.add(Dense(units =32 , kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 

    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) 

    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu')) 

    classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))

    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) 

    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 50)

accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)

mean = accuracies.mean()

variance = accuracies.std()

print("Accuracy mean: "+ str(mean))

print("Accuracy variance: "+ str(variance))