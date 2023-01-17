#Importing libraries

import numpy as np

import pandas as pd

import h5py

import os

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

import keras.backend as k

import tensorflow as tf

from keras.preprocessing import image

from keras.layers import Input, Conv2D, Activation, Dropout, ZeroPadding2D, BatchNormalization,Flatten, Dense, MaxPooling2D

from keras.optimizers import adam 

from keras.models import Model

from sklearn.model_selection import train_test_split

from PIL import Image

from keras.applications.imagenet_utils import preprocess_input

from sklearn.metrics import multilabel_confusion_matrix
# Declaring dataset path

data_path = '../input/'

print(os.listdir(data_path))
# Train and test dataset

train_dataset_path = os.path.join(data_path,'train.csv')

test_dataset_path = os.path.join(data_path, 'test.csv')
#Loading Data

def load_data():

    train_dataset = pd.read_csv(train_dataset_path)

    y_train_orig = train_dataset[['label']]

    X_train_orig = train_dataset.drop(columns = 'label', axis = 1)

    

    X_train_orig = np.array(X_train_orig)

    y_train_orig = np.array(y_train_orig)

    

    test_dataset = pd.read_csv(test_dataset_path)

    X_test_orig =  test_dataset

    X_test_orig =  np.array(X_test_orig)

    

    classes = np.unique(y_train_orig)

    

    return X_train_orig, y_train_orig, X_test_orig, classes
X_train_orig, y_train_orig, X_test_orig, classes = load_data()
print('Shape of X_train_orig = ' + str(X_train_orig.shape))

print('Shape of y_train_orig = ' + str(y_train_orig.shape))

print('Shape of X_test_orig = ' + str(X_test_orig.shape))

print('classes = ' + str(classes))
#Reshaping for each row of data to 4D array 

X_train_orig = np.reshape(X_train_orig, (X_train_orig.shape[0],28,28,1))

X_test_orig = np.reshape(X_test_orig, (X_test_orig.shape[0],28,28,1))



#Normalization

X_train_orig = X_train_orig / 255

X_test_orig  = X_test_orig / 255
print("number of training examples = "   + str(X_train_orig.shape[0]))

print ("number of test examples = " + str(X_test_orig.shape[0]))

print ("X_train shape: " + str(X_train_orig.shape))

print ("y_train shape: " + str(y_train_orig.shape))

print ("X_test shape: " + str(X_test_orig.shape))
# Splitting the train set data to train and test

X_train, X_test, y_train, y_test = train_test_split(X_train_orig,

                                                   y_train_orig,

                                                   test_size = 0.1,

                                                   random_state = 1)
print("number of training examples = "   + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("y_train shape: " + str(y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("y_test shape: " + str(y_test.shape))
# plotting the labels frequency 

def plot_frequnecy(data):

    unique, counts = np.unique(data, return_counts=True)

    print('class is in training set = ' + str(unique))

    print('frequency of classess = ' + str(counts))

    plt.bar(unique, counts)

    plt.title('Count of Labels of Training Set')

    plt.xlabel('Classes')

    plt.ylabel('Count of classes')

    plt.show()
plot_frequnecy(y_train_orig)
# Creating the Model

def digit_recognize_model(input_shape,

                          pad_size,

                          conv_filters0,

                          conv_filters1,

                          conv_kernel_size0,

                          conv_kernel_size1,

                          max_pool_size):

    

    X_input = Input(input_shape)

    

    X = ZeroPadding2D(padding = pad_size)(X_input)

    

    

    X = Conv2D(filters = conv_filters0,

               kernel_size = conv_kernel_size0,

               strides = (1,1),

               name = 'conv1')(X)

    X = BatchNormalization(axis = 3,

                           name = 'bn1')(X)

    X = Activation('relu',

                   name = 'activation1')(X)

    X = Dropout(rate = 0.25,

               name = 'dropout1')(X)

    

    X = MaxPooling2D(pool_size = max_pool_size,

                     name = 'Max_Pool1') (X)

    

    X = Conv2D(filters = conv_filters1,

               kernel_size = conv_kernel_size1,

               strides = (1,1),

               name = 'conv2')(X)

    X = BatchNormalization(axis = 3,

                           name = 'bn2')(X)

    X = Activation('relu',

                  name = 'activation2')(X)    

    X = Dropout(rate = 0.25,

               name = 'dropout2')(X)

    

    X = MaxPooling2D(pool_size = max_pool_size,

                     name = 'Max_Pool2') (X)

    

    X = Flatten(name = 'flatten')(X)

    

    X = Dense(units  = 10,

             activation = 'softmax',

             name = 'fc')(X)

    

    model = Model(inputs = X_input,

                  outputs = X,

                  name = 'Digit Recognizer')

    return model
pad_size = (3,3)

conv_filters0 = 32

conv_filters1 = 10

conv_kernel_size0 = (7,7)

conv_kernel_size1 = (7,7)

max_pool_size = (2,2)

input_shape = X_train.shape[1:]

digit_recognizer_model = digit_recognize_model(input_shape,

                                               pad_size,

                                               conv_filters0,

                                               conv_filters1,

                                               conv_kernel_size0,

                                               conv_kernel_size1,

                                               max_pool_size)
# Checking the summary of the model

digit_recognizer_model.summary()
# Compile the model

digit_recognizer_model.compile(optimizer = adam(lr=0.001,decay=1e-6),

                               loss = 'sparse_categorical_crossentropy',

                               metrics = ['acc'] )
# Fitting the model to train set

history = digit_recognizer_model.fit(X_train,

                                     y_train,

                                     epochs = 40,

                                     batch_size = 50)
# Evaluating the Validating set

score = digit_recognizer_model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0] * 100,'%' )

print('Test accuracy:', score[1] * 100,'%' )
# Predicting the test data

y_pred_1 = digit_recognizer_model.predict(X_test)
# Gmaking label to 1 dimension

y_pred = np.argmax(y_pred_1, axis=1)
# Confusion matrix

cm = multilabel_confusion_matrix(y_test, y_pred)
classification_accuracy = []

error_rate = []

class_num = []
for i in classes:

    correct_predictions = cm[i][0][0] + cm[i][1][1]

    total_predictions = np.sum(cm[i])

    classification_accuracy.append(correct_predictions / total_predictions * 100)

    error_rate.append((1 - (correct_predictions / total_predictions)) * 100)

    class_num.append(i)
print('classification_accuracy : ' + str(sum(classification_accuracy)/10))

print('error_rate : ' + str(sum(error_rate)/10))
# Error rate 

plt.title('Error rate between training and test set')

plt.xlabel('Classes')

plt.ylabel('Error rate')

plt.bar(class_num , error_rate, tick_label = class_num)
# Loss during training set

x = history.history['loss'] 

plt.xlabel('number of epochs')

plt.ylabel('loss')

plt.title('loss on training set during training')

plt.plot(x)
# Accuracy during training set

x = history.history['acc'] 

plt.xlabel('number of epochs')

plt.ylabel('accuracy')

plt.title('Accuracy on training set during training')

plt.plot(x)
# Predicting and showing the image

for i in range(1):

    reshape_to_feed = X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])

    Y_pred = digit_recognizer_model.predict(reshape_to_feed)

    y_pred = np.argmax(Y_pred)

    print('Prediction of '+ str(i) +' is = '+ str(y_pred))

    img = np.reshape(X_test[i],(X_test[i].shape[0],X_test[i].shape[1]))

    plt.imshow(img,cmap='gray')
# Submitting the results

y_predictions = digit_recognizer_model.predict(X_test_orig)

y_predictions = np.argmax(y_predictions, axis = 1)

label = np.array(range(1,len(y_predictions)+1))

submissions = pd.DataFrame({"Image ID": label,

                            "Label": y_predictions})

submissions.to_csv("DigitRecognizer.csv", index=False, header=True)
# to check whether it's updated

sub = pd.read_csv('DigitRecognizer.csv')

sub.head(5)