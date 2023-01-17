import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.layers import concatenate, GlobalAveragePooling2D
%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
def dense_layer(X):
    
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    X_shortcut = X
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid',  kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',  kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = concatenate([X_shortcut, X], axis=3)
    
    return X
def transition_layer(X, filters=128):
    
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    
    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(X)
    
    return X
def dense_block(X, dense_num = 6):
    
    '''X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)'''
    
    for l in range(dense_num):
        X = dense_layer(X)
    
    return X
def DenseNet121(input_shape=(224,224,3), classes=3):
    
    X_input = Input(input_shape)
    
    X = BatchNormalization(axis = 3)(X_input)
    X = Activation('relu')(X)
    X = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(X)
    
    X = dense_block(X, dense_num=6)
    X = transition_layer(X, filters=128)
    X = dense_block(X, dense_num=12)
    X = transition_layer(X, filters=256)
    X = dense_block(X, dense_num=24)
    X = transition_layer(X, filters=512)
    X = dense_block(X, dense_num=16)
    
    X = GlobalAveragePooling2D()(X)
    
    # Output Layer
    X = Dense(1, activation='sigmoid', kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name='DenseNet121')

    return model
model = DenseNet121(input_shape = (224, 224, 3), classes = 2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
IMAGE_SIZE = 224
CHANNELS = 3
DATADIR = '../input/chest-xray-pneumonia/chest_xray/'
test_path = DATADIR + '/test/'
valid_path = DATADIR + '/val/'
train_path = DATADIR + '/train/'
BATCH_SIZE = 16
CATEGORIES = ["NORMAL", "PNEUMONIA"]
if CHANNELS == 1:
    color_mode = "grayscale"
elif CHANNELS == 3:
    color_mode = "rgb"
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1/255)
train_images = train_datagen.flow_from_directory(train_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode='binary',classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)
test_datagen = ImageDataGenerator(rescale=1/255)
test_images = test_datagen.flow_from_directory(test_path,target_size=(IMAGE_SIZE,IMAGE_SIZE),class_mode='binary',classes=CATEGORIES,color_mode=color_mode,batch_size=BATCH_SIZE)
history = model.fit_generator(train_images,validation_data=test_images,epochs=10, steps_per_epoch=len(train_images)/BATCH_SIZE,validation_steps=len(test_images)/BATCH_SIZE)
import matplotlib.pyplot as plt
acc = history.history['loss']
val_acc = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(10,6))
plt.plot(epochs,acc,c="red",label="Training")
plt.plot(epochs,val_acc,c="blue",label="Validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
#model.summary()
#plot_model(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
