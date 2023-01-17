import pandas as pd #to parse csv data

import matplotlib.pyplot as plt # to plot the image

import numpy as np

import tflearn 

import tensorflow as tf

from keras.utils.np_utils import to_categorical #for one hot encoding of training labels
# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

print("Train set has {0[0]} rows and {0[1]} columns".format(train.shape))
#loop through the dataframe line by line and print the image for the arranged data

for x in range(1, 10): #first 10 rows are parsed

    rowData=np.array(train[x-1:x]) #np array rowData contain all 785 values (1 label value+784 pixel value)

    label=np.resize(rowData,(1,1)) #np array label gets the first value from rowdata

    print('label shape             ->',label.shape) #printing the shape of np array label

    print('label                   ->',label.ravel()) #Image label

    rowWithIndex = rowData.ravel()#scalar data with 785 items    

    print('row with index shape    ->',rowWithIndex.shape)

    rowWithOutIndex = rowWithIndex[1:785:1]#scalar image data with 784 pixel values

    print('row without index shape ->',rowWithOutIndex.shape)

    Image1=np.resize(rowWithOutIndex,(28,28)) #28x28 Image

    print('Image shape             ->',Image1.shape) #printing Image shape

    plt.imshow(Image1, interpolation='nearest') #plotting

    plt.show()
# Split data into training set and labels

y_train = train.ix[:,0].values #all input labels, first cloumn(index 0) of each row in the train csv file

trainX = train.ix[:,1:].values #remaining 784 values after(from index 1 till end) the first colum. 



print(y_train.shape)

print(trainX.shape)

#one hot encoded form of labels

y_train_one_hot = to_categorical(y_train)

print(y_train_one_hot)
#DNN - input layer of 784 inputs, 4 hidden layers and a softmax layer at output

def build_model():

    tf.reset_default_graph() 

    net = tflearn.input_data([None, 784]) #input layer with 784 inputs

    net = tflearn.fully_connected(net, 128, activation='ReLU') #hidden layer1

    net = tflearn.fully_connected(net, 64, activation='ReLU') #hidden layer2

    net = tflearn.fully_connected(net, 32, activation='ReLU') #hidden layer3

    net = tflearn.fully_connected(net, 10, activation='softmax') #output layer

    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')

    model = tflearn.DNN(net)

    return model

model = build_model()
#training

model.fit(trainX, y_train_one_hot, validation_set=0.1, show_metric=True, batch_size=300, n_epoch=50)

#inference

testX = test.ix[:,0:].values

def prediction(predictions):

    return np.argmax(predictions,1)

predictions = prediction(model.predict(testX))
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

#submissions.to_csv("submission.csv", index=False, header=True)

print(submissions)