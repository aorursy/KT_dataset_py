#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from tensorflow.keras import datasets,models,layers
%matplotlib inline

# Adding TF Cifar10 Data ..
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


print('X Train Shape is : ' , X_train.shape)
print('X Train  is : ' , X_train[5])
print('---------------------------------------- ')
print('X Test Shape is : ' , X_test.shape)
print('X Test  is : ' , X_test[5])
print('---------------------------------------- ')
print('y Train Shape is : ' , y_train.shape)
print('y Train is : ' , y_train[5])
print('---------------------------------------- ')
print('y Test Shape is : ' , y_test.shape)
print('y Test  is : ' , y_test[5])
print('---------------------------------------- ')
print('All y is : ' , np.unique(y_train))
print('---------------------------------------- ')

# Drawing sample . 
plt.imshow(X_train[5])
# Normalize the data .
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
# STEP 1 : Building the Model 

model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),
    padding='same',activation='relu',
    kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu',kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# STEP 2 : Compiling the Model  

model.compile(optimizer ='adam',loss='categorical_crossentropy',metrics=['accuracy'])



# Step 3 : Training   

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

print('Model Details are : ')
print(model.summary())



# STEP 4 : Predicting   

y_pred = model.predict(X_train)

print('Prediction Shape is {}'.format(y_pred.shape))
print('Prediction items are {}'.format(y_pred[:5]))



# STEP 5 : Evaluating   

ModelLoss, ModelAccuracy = model.evaluate(X_train, y_train)

print('Model Loss is {}'.format(ModelLoss))
print('Model Accuracy is {}'.format(ModelAccuracy ))
def plotmodelhistory(history): 
    fig, axs = plt.subplots(1,2,figsize=(15,5)) 
    # summarize history for accuracy
    axs[0].plot(history.history['accuracy']) 
    axs[0].plot(history.history['val_accuracy']) 
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy') 
    axs[0].set_xlabel('Epoch')
    
    axs[0].legend(['train', 'validate'], loc='upper left')
    # summarize history for loss
    axs[1].plot(history.history['loss']) 
    axs[1].plot(history.history['val_loss']) 
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss') 
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'validate'], loc='upper left')
    plt.show()

# list all data in history
print(history.history.keys())
plotmodelhistory(history)
