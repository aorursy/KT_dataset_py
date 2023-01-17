import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping

import h5py

import numpy as np



h5f = h5py.File('/kaggle/input/street-view-house-nos-h5-file/SVHN_single_grey1.h5', 'r')

h5w = h5py.File('/kaggle/input/cnn-mnist-weights-pretrained/cnn_mnist_weights.h5', 'r')

h5f
X_train = h5f['X_train'][:]

y_train = h5f['y_train'][:]

X_test = h5f['X_test'][:]

y_test = h5f['y_test'][:]



h5f.close() #close this file
print('X_train:', X_train.shape)

print('y_train:', y_train.shape)

print('X_test:', X_train.shape)

print('y_test:', y_train.shape)
import matplotlib.pyplot as plt

%matplotlib inline



fig = plt.figure(figsize = (10,10))



rows = 10

columns = 10

w = 10

h = 10



for i in range(1, rows * columns + 1):

    img = X_test[i]

    fig.add_subplot(rows, columns,i)

    plt.imshow(img, cmap = 'gray')

plt.show()
print(X_train.shape) #before reshape
import cv2

#Create zero array for X_train, X_test

X_train_res = np.zeros((X_train.shape[0], 28,28), dtype = np.float32) #create a zero array of size 28*28 same as MNIST

X_test_res = np.zeros((X_test.shape[0], 28, 28), dtype = np.float32)



for i in range(X_train.shape[0]):

    X_train_res[i,:,:] = cv2.resize(X_train[i], dsize = (28,28), interpolation = cv2.INTER_CUBIC)

    

for i in range(X_test.shape[0]):

    X_test_res[i,:,:] = cv2.resize(X_test[i], dsize = (28,28), interpolation = cv2.INTER_CUBIC)

    

print(X_train_res.shape)

print(X_test_res.shape)
img_rows, img_cols = 28, 28



X_train_CNN = X_train_res.reshape(X_train_res.shape[0], img_rows, img_cols, 1)

X_train_CNN.shape

X_test_CNN = X_test_res.reshape(X_test_res.shape[0], img_rows, img_cols, 1)

print(X_train_CNN.shape)



#Shape of 1 image would be as given below, this would be useful while creating models

input_shape  = (img_rows, img_cols, 1)

print(input_shape)
X_train_CNN = X_train_CNN.astype('float32')

X_test_CNN =  X_test_CNN.astype('float32')



#Normalizing the input

X_train_CNN = X_train_CNN / 255.0

X_test_CNN = X_test_CNN / 255.0



print(X_train_CNN.shape)
y_train
#convert class vectors to binary class metrics

num_classes = 10 # since we will only classify nos between 0-9

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[0]
#Set model hyperparameters

num_classes = 10



#Define the layers of the model

model_CNN = Sequential()



#1. Conv Layer

model_CNN.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape, name = 'Conv1'))



#2. Conv Layer

model_CNN.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = input_shape, name = 'Conv2'))



#3. MaxPooling Layer

model_CNN.add(MaxPooling2D(pool_size = (2,2), name = 'Max1'))



#4. Dropout this prevents model from overfitting

model_CNN.add(Dropout(0.25, name = 'Drop1'))



#5. Flatten Layer

model_CNN.add(Flatten())



#6. Fully Connected Layer

model_CNN.add(Dense(128, activation = 'relu', name = 'Dense1'))



#7. Dropout

model_CNN.add(Dropout(0.5, name = 'Drop2'))



#8. Fully Connected Layer

model_CNN.add(Dense(num_classes, activation = 'softmax', name = 'Dense2'))



model_CNN.summary()


for layer in model_CNN.layers:

    if('Dense' not in layer.name):

        layer.trainable = False

    else:

        layer.trainable = True

        

#Module to output colorful statements

from termcolor import colored



#Check which layers have been frozen

for layer in model_CNN.layers:

    print(colored(layer.name, 'blue'))

    print(colored(layer.trainable, 'red'))

    
model_CNN.load_weights('/kaggle/input/cnn-mnist-weights-pretrained/cnn_mnist_weights.h5')
from keras.optimizers import Adam

from keras.losses import categorical_crossentropy



optimizer = Adam(lr = 0.001)

batch_size = 128

num_classes = 10

epochs = 12



model_CNN.compile(optimizer = optimizer, loss = categorical_crossentropy, metrics = ['accuracy'])
model_CNN.fit(X_train_CNN, y_train,

             batch_size = batch_size,

             epochs = epochs,

             verbose = 1,

             validation_data = (X_test_CNN, y_test))

             #callbacks = [tensorboard_callback, early_stopping, model_checkpoint])
score = model_CNN.evaluate(X_test_CNN, y_test)

print('Test Loss: ', score[0])

print('Test Accuracy: ', score[1])
plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[30].reshape(28,28), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[30].reshape(1,28,28,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[50].reshape(28,28), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[50].reshape(1,28,28,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[78].reshape(28,28), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[78].reshape(1,28,28,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[130].reshape(28,28), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[130].reshape(1,28,28,1))))
model_CNN.save('./cnn_svhn.h5')

model_CNN.save_weights('./cnn_svhn_weights.h5')