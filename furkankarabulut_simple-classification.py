# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load




import numpy as np
import tensorflow as tf
import keras
import pandas as pd
import zipfile
import cv2
import sklearn.model_selection
from keras import backend as K
from keras.utils import to_categorical


#dl libraraies
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# specifically for cnn
from tensorflow.keras.layers import Dropout, Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

data = pd.read_csv('/kaggle/input/game-of-deep-learning-ship-datasets/train/train.csv')
train_images = pd.read_csv('/kaggle/input/game-of-deep-learning-ship-datasets/train/train.csv')
test_images = pd.read_csv('/kaggle/input/game-of-deep-learning-ship-datasets/test_ApKoW4T.csv')
label = np.array(data['category'])
def resize():
    res_img =[]
    train_img = np.array(train_images['image'])
    
    for i in train_img:
        try:
            images = cv2.imread('/kaggle/input/game-of-deep-learning-ship-datasets/train/images/'+i , 1)
            res = cv2.resize(images , (200,150))
            res_img.append(res)
        except Exception as e:
            print(str(e))
    res_img = np.array(res_img)
    return res_img

z = resize()



#Train-Test Split 

label = to_categorical(label)
x_train , x_test , y_train , y_test = sklearn.model_selection.train_test_split(z , label , test_size=0.1)
print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)
print(y_train.shape)
print(y_test.shape)

#Normalization

x_train = x_train/255.0
x_test = x_test/255.0

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150,200,3))

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- we have 6 classes
predictions = Dense(6, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all VGG16 convolutional layers
for layer in base_model.layers:
    layer.trainable = False
    

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally 
        height_shift_range=0.2,  # randomly shift images vertically 
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=32),
          steps_per_epoch=len(x_train) / 32, epochs=50)
model.evaluate(x_test,y_test)