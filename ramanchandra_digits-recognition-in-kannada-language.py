#Import required packages



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle

import scikitplot as skplt

from sklearn.metrics import classification_report,confusion_matrix



#Import packages for Deep learning

import keras

from keras.layers import Conv2D,Dense,Dropout,Flatten,Input

from keras.layers import Activation,MaxPooling2D

from keras.models import Sequential

from keras.optimizers import Adam,sgd,RMSprop,Nadam,Adadelta,Adamax

from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping

from keras.utils import to_categorical,plot_model

from keras.models import Model

from keras import backend as K



#Ensures consistency across runs

from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)

import os

import warnings

warnings.filterwarnings('ignore')

#Network visualization packages

import graphviz as gv

import pydot as pydot;

ds_dir = '/kaggle/input/kannada-mnist/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/Kannada_MNIST'



X_train = np.load(os.path.join(ds_dir,'X_kannada_MNIST_train.npz'))['arr_0']

X_test = np.load(os.path.join(ds_dir,'X_kannada_MNIST_test.npz'))['arr_0']

y_train = np.load(os.path.join(ds_dir,'y_kannada_MNIST_train.npz'))['arr_0']

y_test = np.load(os.path.join(ds_dir,'y_kannada_MNIST_test.npz'))['arr_0']



print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
plt.figure(figsize=(6,6))

plt.imshow(X_train[1],cmap='inferno')

plt.title(y_train[1])
plt.figure(figsize=[10,8])

for i in range(50):

    plt.subplot(5, 10, i+1)

    plt.axis('Off')

    if i < 10:

        plt.title(y_train[i])

    plt.imshow(X_train[i], cmap='inferno')
y_classes=np.unique(y_train)

y_classes
#Reshape the datasets

X_train=X_train.reshape(-1,28,28,1)

X_test=X_test.reshape(-1,28,28,1)

print('Size after reshaping the datasets')

X_train.shape,X_test.shape

#Change datatype from uint8 to float64 & scale input data to the range (0,1)

X_train=X_train.astype('float64')/255

X_test=X_test.astype('float64')/255
#convert target labels from integers to categorical

y_train=to_categorical(y_train,num_classes=10,dtype='uint8')

print('Shape of the target labels...')

print(y_train.shape,y_test.shape)
#Split the train data

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=42)

print('Shape of the train data..')

print(X_train.shape,y_train.shape)

print('Shape of the validation data..')

print(X_val.shape,y_val.shape)
#Shuffle train samples

X_train,y_train=shuffle(X_train,y_train,random_state=42)
#LeNet architecture

class LeNet:

     def build(width,height,depth,classes):

            #Initialize the model

            model=Sequential()

            input_shape=(height,width,depth)

            if K.image_data_format()=='channels_first':

                input_shape=(depth,height,width)

            #first set of Conv=>ReLU=> Pool layers

            model.add(Conv2D(20,(5,5),padding='same',input_shape=input_shape))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

            #Second set of Conv=>ReLU=> Pool layers

            model.add(Conv2D(50,(5,5),padding='same',input_shape=input_shape))

            model.add(Activation('relu'))

            model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

            #FC=>ReLU layer with 500 nodes

            model.add(Flatten())

            model.add(Dense(500))

            model.add(Activation('relu'))

            #Add dense & softmax layer

            model.add(Dense(classes))

            model.add(Activation('softmax'))

            return model
#Initialize the optimizer & model

opt=sgd(lr=0.01)

model=LeNet.build(width=28,height=28,depth=1,classes=10)

plot_model(model,to_file='LeNet.png',show_shapes=True)

model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['acc'])

model.summary()
#Visualize LeNet architecture using graphviz

from IPython.display import Image

Image('LeNet.png')

#Prepare the model & save in the directory

save_dir=os.path.join(os.getcwd(),'saved_models')

model_name='Kannada_digit_recognition_model.h5'

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

filepath=os.path.join(save_dir,model_name)



#Prepare callbacks for model saving & for learning rate adjustment

checkpoint=ModelCheckpoint(filepath=filepath,

                          monitor='val_acc',

                          verbose=1, mode='max',

                          save_best_only=True)

early_stopping=EarlyStopping(monitor='val_acc',mode='max',

                            verbose=1,patience=20)                      

callbacks=[checkpoint,early_stopping]
#train the network

print('training the network....')

history=model.fit(X_train,y_train,batch_size=32,

                   validation_data=(X_val,y_val),epochs=10,

                   verbose=1,

                    callbacks=callbacks,

                    shuffle=True

                   )
#Evaluate the network

print('Evaluatng the network...')

y_pred=model.predict(X_test,batch_size=32)

target_names=['0','1','2','3','4','5','6','7','8','9']

classification_report=classification_report(y_test,y_pred.argmax(axis=1),

                                            target_names=target_names)

print('Classification Report...')

print(classification_report)
#plot our confusion matrix

skplt.metrics.plot_confusion_matrix(y_test,y_pred.argmax(axis=1),normalize=False,figsize=(12,8))

plt.show()
#Plot the training loss & accuracy

plt.style.use('ggplot')

plt.figure()

plt.plot(np.arange(0,10),history.history['loss'],label='train_loss')

plt.plot(np.arange(0,10),history.history['val_loss'],label='validatin_loss')

plt.plot(np.arange(0,10),history.history['acc'],label='train_accuracy')

plt.plot(np.arange(0,10),history.history['val_acc'],label='validatin_accuracy')

plt.title('Training loss & Accuracy')

plt.xlabel('#epochs')

plt.ylabel('Loss/Accuracy')

plt.legend()

plt.show()