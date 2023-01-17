import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.callbacks import ModelCheckpoint, EarlyStopping

!pip install tensorboardcolab
import h5py

import numpy as np



h5f = h5py.File('/kaggle/input/street-view-house-nos-h5-file/SVHN_single_grey1.h5', 'r')

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

img_rows, img_cols = 32,32 #capturing this separately to be used later 
#For Fully Connected Network

X_train_FC = X_train.reshape(X_train.shape[0], img_rows * img_cols) #32*32

X_test_FC = X_test.reshape(X_test.shape[0], img_rows * img_cols)

print(X_train_FC.shape)

#For Convolutional Neural Network

X_train_CNN = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_train_CNN.shape

X_test_CNN = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

print(X_train_CNN.shape)



#Shape of 1 image would be as given below, this would be useful while creating models

input_shape  = (img_rows, img_cols, 1)

print(input_shape)
X_train_FC[0]
X_train_CNN[0]
#For FC Network

X_train_FC = X_train_FC.astype('float32')

X_test_FC =  X_test_FC.astype('float32')



#Normalizing the input

X_train_FC = X_train_FC / 255.0

X_test_FC = X_test_FC / 255.0



print(X_train_FC.shape)
X_train_FC.max() #This is to cross check whether inputs have been normalized.
#For CNN 

X_train_CNN = X_train_CNN.astype('float32')

X_test_CNN =  X_test_CNN.astype('float32')



#Normalizing the input

X_train_CNN = X_train_CNN / 255.0

X_test_CNN = X_test_CNN / 255.0



print(X_train_CNN.shape)
X_train_CNN.max() #This is to cross check whether inputs have been normalized.
y_train
#convert class vectors to binary class metrics

num_classes = 10 # since we will only classify nos between 0-9

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)

y_train[0]
y_test[0]
model_FC = Sequential()

model_FC.add(Dense(100, input_shape = (1024, ), activation = 'relu')) #hidden layer

model_FC.add(Dense(10, activation = 'softmax')) #output layer

model_FC.summary()
model_FC.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_FC.fit(X_train_FC, y_train, batch_size = 128, epochs = 10, validation_data = (X_test_FC, y_test))
plt.figure(figsize = (2,2))

plt.imshow(X_test_FC[5000].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_FC.predict(X_test_FC[5000].reshape(1,1024))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_FC[9876].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_FC.predict(X_test_FC[9876].reshape(1,1024))))
#Set model hyperparameters

num_classes = 10



#Define the layers of the model

model_CNN = Sequential()



#1. Conv Layer

model_CNN.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))



#2. Conv Layer

model_CNN.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', input_shape = input_shape))



#3. MaxPooling Layer

model_CNN.add(MaxPooling2D(pool_size = (2,2)))



#4. Dropout this prevents model from overfitting

model_CNN.add(Dropout(0.25))



#5. Flatten Layer

model_CNN.add(Flatten())



#6. Fully Connected Layer

model_CNN.add(Dense(128, activation = 'relu'))



#7. Dropout

model_CNN.add(Dropout(0.5))



#8. Fully Connected Layer

model_CNN.add(Dense(num_classes, activation = 'softmax'))



model_CNN.summary()
model_CNN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#Initializing the tensorboard callback for visualization of training



#For COLAB

#Import tensorboard colab modules for creating a tensorboard call back which will pass in model.fit function



#from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback

#from time import time



#Tensorboard callback is going to be added to model.fit function to draw graphs of loss values after every epoch

#tbc = TensorBoardColab()



#For KAGGLE

# Load the extension and start TensorBoard



%load_ext tensorboard

%tensorboard --logdir logs

import tensorflow as tf

tensorboard_callback = tf.keras.callbacks.TensorBoard("logs")
#Adding Early Stopping function to the Fit function is going to stop the training. 

#That is, when the validation loss doesn't change even by '0.001' for more than 10 continuous epochs



early_stopping = EarlyStopping(monitor = 'val_loss', min_delta = 0.001, patience = 10)



#Adding Model Checkpoint callback to the fit function is going to save the weights whenever the val_loss achieves a new low value



model_checkpoint = ModelCheckpoint('svhn_cnn_checkpoint_{epoch:02d}_loss{val_loss:.4f}.h5',

                                  monitor = 'val_loss',

                                  verbose = 1,

                                  save_best_only = True,

                                  save_weights_only = True,

                                  mode = 'auto',

                                  period = 1)
model_CNN.fit(X_train_CNN, y_train,

             batch_size = 128,

             epochs = 10,

             verbose = 1,

             validation_data = (X_test_CNN, y_test))

             #callbacks = [tensorboard_callback, early_stopping, model_checkpoint])
score = model_CNN.evaluate(X_test_CNN, y_test)

print('Test Loss: ', score[0])

print('Test Accuracy: ', score[1])
plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[30].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[30].reshape(1,32,32,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[50].reshape(32,32), cmap = 'gray')

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[50].reshape(1,32,32,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[100].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[100].reshape(1,32,32,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[230].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[230].reshape(1,32,32,1))))



plt.figure(figsize = (2,2))

plt.imshow(X_test_CNN[1000].reshape(32,32), cmap = 'gray') #image, reshape size, cmap

plt.show()

print(np.argmax(model_CNN.predict(X_test_CNN[1000].reshape(1,32,32,1))))
model_CNN.save('./cnn_svhn.h5')

model_CNN.save_weights('./cnn_svhn_weights.h5')