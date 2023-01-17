import numpy as np

import pandas as pd

from keras import backend as K

K.set_image_dim_ordering('th')

import matplotlib.pyplot as plt

%matplotlib inline
# load training data

train_data = pd.read_csv('../input/train.csv')

print("Training data shape: "+ str(train_data.shape))

train_data.head()
# load test data

test_data = pd.read_csv('../input/test.csv')

print("Test data shape: " + str(test_data.shape))

test_data.head()
#Get train data pixel values and labels (i.e. digits)

X_train = (train_data.iloc[:,1:].values).astype('float32') # pixel values(the columns with heading pixel 0 to pixel 783 above )

Y_train = train_data.iloc[:,0].values.astype('int32') # target labels(the column with heading "label" above)



#Get test data pixel values 

X_test = test_data.values.astype('float32')
#Convert train data to format - (num_images, img_rows, img_cols)  

X_train = X_train.reshape(X_train.shape[0], 28, 28)



#lets Plot a train image

plt.imshow(X_train[7], cmap=plt.get_cmap('gray'))

plt.title(Y_train[7]);
#For colour channel gray, we need to expand one more dimension as 1 for both X_train and X_test

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)



print("X_train data shape: "+ str(X_train.shape))

print("X_test data shape: " + str(X_test.shape))
#Calculate Mean of training data

mean_x = X_train.mean().astype(np.float32)

#Calculate Std deviation of training data

std_x = X_train.std().astype(np.float32)
def standardize(x):

    '''

    Function to subtract mean from x to zero center x. 

    Then divide the zero centered x by std deviation of x.

    '''

    return (x-mean_x)/std_x
#Stadardise train data

X_train = standardize(X_train)
#ONE HOT ENCODING OF LABELS

from keras.utils import np_utils



Y_train= np_utils.to_categorical(Y_train)

num_classes = Y_train.shape[1]

print(num_classes)
# fix random seed 

seed = 43

np.random.seed(seed)
from sklearn.model_selection import train_test_split



# divide data into training and validation set

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.15, random_state=80)



print('train_features shape: ', X_train.shape)

print('vali_features shape: ', X_validation.shape)

print('train_labels shape: ', Y_train.shape)

print('vali_labels shape: ', Y_validation.shape)
from keras.models import Sequential

from keras.layers import Activation, Flatten, Reshape, Dense, Dropout

from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



# hyperparameters

epochs = 5

batch_size = 256 #Preferrably keep it a multiple fo 2



# build the model

num_classes = 10

model = Sequential()

#MODEL 1: Following model gives val_acc: 0.9875

#model.add(Conv2D(64, (5, 5), input_shape=(1, 28, 28), activation='relu', bias_initializer='RandomNormal'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, (5, 5), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())

#model.add(Dense(128, activation='relu'))

#model.add(Dropout(0.2))

#model.add(Dense(32, activation='relu'))

#model.add(Dropout(0.2))

#model.add(Dense(num_classes, activation='softmax'))



#MODEL 2: Following model gives val_acc: 0.9894

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(1, 28, 28), bias_initializer='RandomNormal'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))
#Lets print the model we have created!

print(model.summary())
from keras.optimizers import Adam



model.compile(optimizer='adam',  #Updates the network

              loss='categorical_crossentropy', #Measure of how good network is doing

               metrics=['accuracy']) #Monitor for performance of network
# training

training = model.fit(X_train, Y_train,

                     validation_data=(X_validation, Y_validation),

                     epochs=epochs,

                     batch_size=batch_size, 

                     verbose=1)
#Lets plot loss and accuracy for training as well as validation!

def plot_model_history(model_history):

    '''

    Function to plot trainging and validation loss and accuracy in same row

    input: model_history, Model History

    '''

    fig, axs = plt.subplots(1,2,figsize=(10,5))

    # Accuracy for both training and validation

    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])

    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy')

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['trainining', 'validation'], loc='best')

    # Loss for both training and validation

    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])

    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss')

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['trainining', 'validation'], loc='best')

    plt.show()
plot_model_history(training)
# ZERO MEAN and UNIT STD DEV for test images as well

X_test = standardize(X_test)



# Lets predict labels for test set!

Y_prediction = model.predict(X_test, batch_size=batch_size, verbose=1)



# Pick the predicted class with highest probability to convert from probabilities to digits!!

Y_prediction_digits = np.argmax(Y_prediction, axis=1)
submission = pd.DataFrame({'Label': Y_prediction_digits})

submission.index += 1

submission.index.name = "ImageId"

submission.to_csv('submission_public.csv')