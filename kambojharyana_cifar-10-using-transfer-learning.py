# load the standart libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Import keras functions

from keras import Sequential

from keras.applications import VGG19,ResNet50

'Import the datagenerator to augment images'
from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau

from keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout

'Import to_categorical from the keras utils package to one hot encode the labels'
from keras.utils import to_categorical
#Import dataset
from keras.datasets import cifar10

# load the dataset into train and test from cifar10 object
(X_train,y_train),(X_test,y_test)=cifar10.load_data()
# view the dimensions of train and test data to ensure everything is going fine 
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
num_classes = 10
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test,num_classes)
# checking the dimensions again, to verify the change
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)
# the dimensions of the labels have now changed. 
train_generator = ImageDataGenerator(
                                    rotation_range=2, 
                                    horizontal_flip=True,
                                    zoom_range=.1 )


#now fit it to train data 
train_generator.fit(X_train)
# will be using VGG19, make sure the input shape is same as that of CIFAR10
base_model = VGG19(include_top=False,weights='imagenet',input_shape=(32,32,3),classes=y_train.shape[1])
# use the base model, and add more layers to it. 
model= Sequential()
model.add(base_model) #Adds the base model (in this case vgg19 to model)
model.add(Flatten()) 

# add some dense layers 
model.add(Dense(1024,activation=('relu')))
model.add(Dense(512,activation=('relu'))) 
model.add(Dense(256,activation=('relu'))) 
model.add(Dropout(.3)) #Adding a dropout layer that will randomly drop 30% of the weights
model.add(Dense(128,activation=('relu')))
model.add(Dropout(.2)) #Adding a dropout layer that will randomly drop 20% of the weights
model.add(Dense(10,activation=('softmax'))) #This is the classification layer

# model summay 
model.summary()
# compile the model 
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
batch_size = 64
epochs = 50
history = model.fit_generator(train_generator.flow(X_train,y_train,batch_size=batch_size),
                      epochs=epochs,
                      steps_per_epoch=X_train.shape[0]//batch_size,
                      validation_data=(X_test,y_test))
model.evaluate(X_test,y_test)