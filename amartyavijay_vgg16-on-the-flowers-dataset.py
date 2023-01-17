# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile
import shutil
import tensorflow as tf
import os
import random
from shutil import copyfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
base_dir='/tmp/flower_class'
os.mkdir(base_dir)
training_dir=os.path.join(base_dir,'training')
os.mkdir(training_dir)
validation_dir=os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
print(os.listdir(base_dir))
train_dandelion=os.path.join(training_dir,'dandelion')
os.mkdir(train_dandelion)
train_daisy=os.path.join(training_dir,'daisy')
os.mkdir(train_daisy)
train_rose=os.path.join(training_dir,'rose')
os.mkdir(train_rose)
train_tulip=os.path.join(training_dir,'tulip')
os.mkdir(train_tulip)
train_sunflower=os.path.join(training_dir,'sunflower')
os.mkdir(train_sunflower)
print(os.listdir(training_dir))
validation_dandelion=os.path.join(validation_dir,'dandelion')
os.mkdir(validation_dandelion)
validation_daisy=os.path.join(validation_dir,'daisy')
os.mkdir(validation_daisy)
validation_rose=os.path.join(validation_dir,'rose')
os.mkdir(validation_rose)
validation_tulip=os.path.join(validation_dir,'tulip')
os.mkdir(validation_tulip)
validation_sunflower=os.path.join(validation_dir,'sunflower')
os.mkdir(validation_sunflower)
print(os.listdir(validation_dir))

flower_dandelion_source='../input/flowers-recognition/flowers/dandelion'
flower_daisy_source='../input/flowers-recognition/flowers/daisy'
flower_rose_source='../input/flowers-recognition/flowers/rose'
flower_tulip_source='../input/flowers-recognition/flowers/tulip'
flower_sunflower_source='../input/flowers-recognition/flowers/sunflower'
print('total no. of tulips are',len(os.listdir(flower_tulip_source)))
print('total no. of roses are',len(os.listdir(flower_rose_source)))
print('total no. of daisies are',len(os.listdir(flower_daisy_source)))
print('total no. of sunflowers are',len(os.listdir(flower_sunflower_source)))
print('total no. of dandelions are',len(os.listdir(flower_dandelion_source)))
tulip_files=os.listdir(flower_tulip_source)
print(tulip_files[3:8])
a=0
def split_data(source,training,testing,split_size):

    listall=os.listdir(source)
    #print(len(listall))
    
    for i in listall:
        
        if  os.path.getsize(os.path.join(source,i)) !=0:
            a=2
        else:
            listall.pop(i)

    listall=random.sample(listall, len(listall))
    #print(len(listall))
    b=len(listall)*split_size
    b=round(b)
    #print(b)
    trainall=listall[0:b]
    valall=listall[b:]
    #print(len(valall))
    [copyfile(source + '/' + i, training + '/'+i) for i in trainall]
    [copyfile(source + '/'+i, testing +'/'+i) for i in valall]
    #for i in listall:
        #if listall.index(i)<b:
            #copyfile(source + '/' + i, training + '/'+i)
        #else:
             #copyfile(source + '/'+i, testing +'/'+i)
split_size=0.9            
split_data(flower_dandelion_source,train_dandelion,validation_dandelion,split_size)                
split_data(flower_daisy_source,train_daisy,validation_daisy,split_size)                
split_data(flower_rose_source,train_rose,validation_rose,split_size)                
split_data(flower_tulip_source,train_tulip,validation_tulip,split_size)                
split_data(flower_sunflower_source,train_sunflower,validation_sunflower,split_size)                
print(len(os.listdir(validation_sunflower)))
print(len(os.listdir(train_sunflower)))
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from tensorflow.keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
%matplotlib inline
def identity_block(X, f, filters, stage, block):
     
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
   
    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
     
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
  
    return X

def ResNet50(input_shape=(150, 150, 3), classes=5):
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D((2,2), name="avg_pool")(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model
model = ResNet50(input_shape = (150, 150, 3), classes = 5)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import RMSprop
base_model =ResNet50(input_shape=(150,150,3),
                                               include_top=False,
                                               weights=None)
base_model.trainable = False
base_model.summary()    

last_layer = base_model.get_layer('conv5_block3_out')
print('last layer output shape: ', last_layer.output_shape)
last_output =last_layer.output 
x = layers.Flatten()(last_output)
x = layers.Dense(720,activation='relu')(x)
#x = Dropout(0.3)(x)
x = layers.Dense(5,activation='softmax')(x)   
model = Model(base_model.input,x) 
model.summary()

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss ='categorical_crossentropy' , 
              metrics = ['accuracy'])


from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop
base_model =VGG16(input_shape=(150,150,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
base_model.summary()    

last_layer = base_model.get_layer('block4_pool')
print('last layer output shape: ', last_layer.output_shape)
last_output =last_layer.output 
x = layers.Flatten()(last_output)
x = layers.Dense(720,activation='relu')(x)
x = layers.Dense(5,activation='softmax')(x)   
model = Model(base_model.input,x) 
model.summary()

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss ='categorical_crossentropy' , 
              metrics = ['accuracy'])

TRAINING_DIR =training_dir 
train_datagen =ImageDataGenerator(rescale=1./255,
                                 rotation_range=10,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.1,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=14,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
VALIDATION_DIR = validation_dir
validation_datagen = ImageDataGenerator(rescale=1./255)


validation_generator =validation_generator =validation_datagen.flow_from_directory(VALIDATION_DIR ,
                                                    batch_size=15,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))
history = model.fit_generator(train_generator,
                              epochs=15,
                              verbose=1,
                              validation_data=validation_generator)
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc))
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.title('Training and validation loss')