import pandas as pd
import numpy as np
import os,cv2
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam,SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping, Callback, TensorBoard
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout
from keras.models import Sequential,Input,Model

from keras.initializers import *
%matplotlib inline
batch_size = 32
num_classes = 10
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
train_images.shape,test_labels.shape,test_images.shape,test_labels.shape
fig, ax = plt.subplots(1,2,figsize=(15,5))

sns.countplot(train_labels.ravel(),ax=ax[0])
sns.countplot(test_labels.ravel(),ax=ax[1])

ax[0].set_title("Training Data")
ax[1].set_title("Testing Data");
train_images = train_images.astype("float")
test_images = test_images.astype("float")

train_labels = to_categorical(train_labels,num_classes)
test_labels = to_categorical(test_labels,num_classes)
train_images[0].shape
from IPython.display import clear_output

class DrawingPlot(Callback):
    
    def on_train_begin(self,logs={}):
        
        self.loss = []
        self.val_loss = []
        self.accuracy = []
        self.val_accuracy = []
        self.logs = []
        pass
    
    def on_epoch_end(self,epoch,logs={}):
        
        self.logs.append(logs)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        
        fig, ax = plt.subplots(1,2,figsize=(15,5))
        ax[0].set_title('Loss')
        ax[1].set_title("Accuracy")
        ax[0].plot(self.loss,label='Train Loss')
        ax[0].plot(self.val_loss,label='Test loss')
        ax[1].plot(self.accuracy,label='Train Accuracy')
        ax[1].plot(self.val_accuracy,label='Test Accuracy')
        
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='lower right')
        
        plt.show()
        pass
    
plot = DrawingPlot()
def show_final_history(history):
    
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].set_title('Loss')
    ax[1].set_title("Accuracy")
    ax[0].plot(history.history['loss'],label='Train Loss')
    ax[0].plot(history.history['val_loss'],label='Test loss')
    ax[1].plot(history.history['accuracy'],label='Train Accuracy')
    ax[1].plot(history.history['val_accuracy'],label='Test Accuracy')
    
    ax[0].legend(loc='upper right')
    ax[1].legend(loc='lower right')
    plt.show()
    pass
train_datagen = ImageDataGenerator(rescale=1./255,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  rotation_range=20)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x=train_images,
                                   y=train_labels,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   seed=42)

test_generator = test_datagen.flow(x=test_images,
                                 y=test_labels,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 seed=42)
def identity_block(X,f,filters,stage,block):
    
    conv_name_base = 'res_'+str(stage)+block+'_branch'
    bn_name_base = 'bn_'+str(stage)+block+'_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    # First Component of Main Path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(1,1),
               padding='same',name=conv_name_base+'2a',
               kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of Main Path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_name_base+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of Main Path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_name_base+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_name_base+'2c')(X)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    pass
def convolutional_block(X,f,filters,stage,block,s=2):
    
    conv_base_name = 'res_' + str(stage) + block + '_branch'
    bn_base_name = 'bn_' + str(stage) + block + '_branch'
    
    F1,F2,F3 = filters
    
    X_shortcut = X
    
    ### MAIN PATH ###
    # First component of main path
    X = Conv2D(filters=F1,kernel_size=(3,3),strides=(s,s),
              padding='same',name=conv_base_name+'2a',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2a')(X)
    X = Activation('relu')(X)
    
    # Second Component of main path
    X = Conv2D(filters=F2,kernel_size=(f,f),strides=(1,1),
              padding='same',name=conv_base_name+'2b',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2b')(X)
    X = Activation('relu')(X)
    
    # Third Component of main path
    X = Conv2D(filters=F3,kernel_size=(3,3),strides=(1,1),
              padding='same',name=conv_base_name+'2c',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name=bn_base_name+'2c')(X)
    
    # Shortcut path
    X_shortcut = Conv2D(filters=F3,kernel_size=(1,1),strides=(s,s),
                       padding='same',name=conv_base_name+'1',
                       kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(name=bn_base_name+'1')(X_shortcut)
    
    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    return X
    pass
def ResNet(input_shape,classes):
    
    X_input = Input(input_shape)
    
    # Zero Padding
    X = ZeroPadding2D((3,3))(X_input)
    
    # Stage 1
    X = Conv2D(8,(7,7),strides=(2,2),name='conv1',kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3),strides=(2,2))(X)
    X = Dropout(0.25)(X)
    
    # Stage 2
    X = convolutional_block(X,f=3,filters=[16,16,32],stage=2,block='A',s=1)
    X = identity_block(X,3,[16,16,32],stage=2,block='B')
    X = identity_block(X,3,[16,16,32],stage=2,block='C')
    X = Dropout(0.25)(X)
    
    # Stage 3
    X = convolutional_block(X,f=3,filters=[32,32,64],stage=3,block='A',s=2)
    X = identity_block(X,f=3,filters=[32,32,64],stage=3,block='B')
    X = identity_block(X,f=3,filters=[32,32,64],stage=3,block='C')
    X = identity_block(X,f=3,filters=[32,32,64],stage=3,block='D')
    X = Dropout(0.25)(X)
    
    # Stage 4
    X = convolutional_block(X,f=3,filters=[64,64,128],stage=4,block='A',s=2)
    X = identity_block(X,f=3,filters=[64,64,128],stage=4,block='B')
    X = identity_block(X,f=3,filters=[64,64,128],stage=4,block='C')
    X = identity_block(X,f=3,filters=[64,64,128],stage=4,block='D')
    X = identity_block(X,f=3,filters=[64,64,128],stage=4,block='E')
    X = identity_block(X,f=3,filters=[64,64,128],stage=4,block='F')
    X = Dropout(0.25)(X)
    
    # Stage 5
    X = convolutional_block(X,f=3,filters=[128,128,256],stage=5,block='A',s=1)
    X = identity_block(X,f=3,filters=[128,128,256],stage=5,block='B')
    X = identity_block(X,f=3,filters=[128,128,256],stage=5,block='C')
    X = Dropout(0.25)(X)
    
    # Stage 6
    X = convolutional_block(X,f=3,filters=[256,256,512],stage=6,block='A',s=2)
    X = identity_block(X,f=3,filters=[256,256,512],stage=6,block='B')
    X = identity_block(X,f=3,filters=[256,256,512],stage=6,block='C')
    X = identity_block(X,f=3,filters=[256,256,512],stage=6,block='D')
    X = Dropout(0.25)(X)
    
    # Average Pool Layer
    X = AveragePooling2D((1,1),name="avg_pool")(X)
    
    # Output layer
    X = Flatten()(X)
    X = Dense(classes,activation='softmax',name='fc'+str(classes),
              kernel_initializer=glorot_uniform(seed=0))(X)
    
    model = Model(inputs=X_input,outputs=X,name='ResNet')
    
    return model
    pass
model = ResNet(input_shape=(32,32,3),classes=num_classes)
plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.summary()
opt = Adam(lr=0.0001)

model.compile(optimizer=opt,loss=['categorical_crossentropy'],metrics=['accuracy'])
checkpoint = ModelCheckpoint("model_weights.h5",monitor="val_accuracy",verbose=1,save_best_only=True,
                            mode="max")
tensorboard_callback = TensorBoard("logs")
epochs = 200

history = model.fit_generator(generator = train_generator,
                              steps_per_epoch = train_generator.n//batch_size,
                              epochs = epochs,
                              validation_data = test_generator,
                              validation_steps = test_generator.n//batch_size,
                              callbacks = [checkpoint,tensorboard_callback],
                              verbose = 1)
show_final_history(history)