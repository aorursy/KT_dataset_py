import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import os





import tensorflow as tf

import tflearn 

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected

from tflearn.layers.estimator import regression
Img_size = 28

LR = 1e-3

Model_name = "Fashion_TEST"

Correct_Pred = 0

EPOCH = 10
def Import_data_set():

    #Importing Data Set

    data_train = pd.read_csv('../input/fashion-mnist_train.csv')

    data_test = pd.read_csv('../input/fashion-mnist_test.csv')

    

    #Loading X and y

    X_Train = data_train.iloc[:,1:].values

    Y_Train = data_train.iloc[:,0:1].values

    

    X_Pred = data_test.iloc[:,1:].values

    Y_Pred_Actual = data_test.iloc[:,0:1].values



    #Encording to OneHotArray

    from sklearn.preprocessing import OneHotEncoder

    onehotencoder_y = OneHotEncoder(categorical_features= [0])

    Y_Train = onehotencoder_y.fit_transform(Y_Train).toarray()

    Y_Pred_Actual = onehotencoder_y.transform(Y_Pred_Actual).toarray()

    

    #Reshaping X to 28x28

    X_Train = X_Train.reshape(-1,Img_size,Img_size,1)

    X_Pred = X_Pred.reshape(-1,Img_size,Img_size,1)

    

    from sklearn.cross_validation import train_test_split

    X_train,X_test,Y_train,Y_test = train_test_split(X_Train, Y_Train, test_size = 0.25, random_state = 32)

    

    return(X_train,X_test,Y_train,Y_test,X_Pred,Y_Pred_Actual)

    

    

X_train,X_test,Y_train,Y_test,X_Pred,Y_Pred_Actual = Import_data_set()
#Visulizing Img

'''

im = plt.imshow(X_train[0], cmap='gray')

plt.show()

'''
def Train_Model():

    tf.reset_default_graph()

    #Input for CNN

    convnet = input_data(shape=[None, Img_size, Img_size,1], name='input')

    

    convnet = conv_2d(convnet, 32, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)

    

    convnet = conv_2d(convnet, 64, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)

    

    convnet = conv_2d(convnet, 128, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)

    

    convnet = conv_2d(convnet, 64, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)

    

    convnet = conv_2d(convnet, 32, 5, activation='relu')

    convnet = max_pool_2d(convnet, 5)

    

    convnet = fully_connected(convnet, 1024, activation='relu')

    convnet = dropout(convnet, 0.8)

    

    convnet = fully_connected(convnet, 10, activation='softmax')

    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    

    model = tflearn.DNN(convnet, tensorboard_dir='LOG')

    

    #Loading Previous Models If exsist

    if os.path.exists('{}.meta'.format(Model_name)):

        model.load(Model_name)

        print("Model Loaded")

        

    else:

        print("New Model")  



         

    #Fitting the Model

    model.fit({'input': X_train}, {'targets':Y_train}, n_epoch=EPOCH, validation_set=({'input': X_test}, {'targets': Y_test}), 

        snapshot_step=500, show_metric=True, run_id=Model_name)

    

    

    #Saving the The trained Model

    model.save(Model_name)

    

    return model









model = Train_Model()
#predicting From the Model

Y_Pred = np.round(model.predict(X_Pred))



#Confustion Matrix

from sklearn.metrics import confusion_matrix

for i in range(Y_Pred.shape[1]):

    Correct_Pred+=(confusion_matrix(Y_Pred_Actual[:,i], Y_Pred[:,i])[1][1])

    

#Printing Accuracy of prediction

print("Accuracy = {}%".format(Correct_Pred/100))