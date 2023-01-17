# Import necessary modules

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline





import itertools

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend as K

from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from tensorflow.python.keras.utils.vis_utils import model_to_dot

from tensorflow.python.keras.applications.vgg16 import VGG16

from tensorflow.python.keras.applications.resnet50 import ResNet50,preprocess_input

from sklearn.model_selection import train_test_split,KFold, cross_val_score, GridSearchCV

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array

from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,BatchNormalization,Dropout,Conv2D,MaxPool2D



sns.set(style = "white", context = "notebook", palette = "deep")
# Preprocessing datasets

# 1.Load datasets

os.listdir("../input/Kannada-MNIST")

train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")



# 2.Delete label "label" and "id" in order to reshaping

Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_test = test.drop(labels = ["id"],axis = 1)



# 3.Reshape each flatten image into 28x28 square image and make the train label into categorical matrix

X_train = X_train.values.reshape(-1,28,28)

X_test = X_test.values.reshape(-1,28,28)

Y_train = to_categorical(Y_train,num_classes = 10)

del train,test



# 4.In order to use Resnet-50, we need transfer 28x28 image into 32x32 image, here we expand the each border of iamge

#   with the np.r_ and np.c_ .

new_Cols = np.zeros([28,2])

new_Rows = np.zeros([2,32])

expand_X_train = np.zeros([len(X_train),32,32])

expand_X_test = np.zeros([len(X_test),32,32])



for i in range(len(X_train)):

    X_temp = np.c_[new_Cols,X_train[i]]

    X_temp = np.c_[X_temp,new_Cols]

    X_temp = np.r_[new_Rows,X_temp]

    expand_X_train[i] = np.r_[X_temp,new_Rows]

    

for i in range(len(X_test)):

    X_temp = np.c_[new_Cols,X_test[i]]

    X_temp = np.c_[X_temp,new_Cols]

    X_temp = np.r_[new_Rows,X_temp]

    expand_X_test[i] = np.r_[X_temp,new_Rows]

    

# 5.In order to use Resnet-50, I added 3 channels into every image

new_X_train = np.zeros([len(X_train),32,32,3])

new_X_test = np.zeros([len(X_test),32,32,3])



X_train = expand_X_train.reshape(-1,32,32,1)

X_test = expand_X_test.reshape(-1,32,32,1)



new_X_train[:,:,:,] = X_train

new_X_test[:,:,:,] = X_test
# 6.Split the train and validation data from the orignal train datasets 

random_seed = 2019

X_train,X_val,Y_train,Y_val = train_test_split(new_X_train,Y_train,test_size=0.15,random_state = random_seed)

# Build the model with Resnet-50, an outer Dataset:resnet50 should be added before

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = Sequential()

model.add(ResNet50(include_top=False,input_tensor=None,input_shape=(32,32,3),pooling='avg',classes=10,weights=resnet_weights_path))

model.add(Flatten())

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Dense(10, activation='softmax'))

# Compile the model and Set some parameters.

model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.7)

batch_size=256

epochs=30
# With data augmentation to prevent overfitting

# datagen = ImageDataGenerator(

#         featurewise_center=False,  # set input mean to 0 over the dataset

#         samplewise_center=False,  # set each sample mean to 0

#         featurewise_std_normalization=False,  # divide inputs by std of the dataset

#         samplewise_std_normalization=False,  # divide each input by its std

#         zca_whitening=False,  # apply ZCA whitening

#         rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

#         zoom_range = 0.1, # Randomly zoom image 

#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

#         horizontal_flip=False,  # randomly flip images

#         vertical_flip=False)  # randomly flip images



# datagen.fit(X_train)
# Fit the model

# history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size = batch_size),

#                               epochs = epochs, validation_data = (X_val,Y_val),

#                               verbose = 1, steps_per_epoch = X_train.shape[0] // batch_size

#                               , callbacks=[red_lr])  
# Fit the model

history = model.fit(X_train,Y_train, batch_size = batch_size,epochs = epochs, validation_data = (X_val,Y_val),verbose = 1, callbacks=[red_lr])  
# Predict the label of the test datasets

Y_pred = model.predict(new_X_test)

Y_pred = np.argmax(Y_pred,axis=1)
# Make a submission

sample_sub = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

sample_sub['label'] = Y_pred

sample_sub.to_csv('submission.csv',index=False)