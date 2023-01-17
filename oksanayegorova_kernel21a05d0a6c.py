# tensorflow.python.keras <-> keras

import sys

# !!!  path to multilabel_CM.py for multilable confussion matrix !!!

# sys.path.append('C:\\Users\\Kudin\\Google Диск')

# print(sys.path)

import os

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from skimage import io, filters, color, segmentation

from skimage.transform import rescale, resize, downscale_local_mean, rotate

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np # linear algebra

from keras import losses, models, optimizers

from keras.models import Sequential

from keras.layers import (Convolution2D, Dense, Dropout, GlobalAveragePooling2D, 

                              GlobalMaxPool2D, Input, MaxPool2D, concatenate, Activation,  

                              MaxPooling2D,Flatten,BatchNormalization, Conv2D,AveragePooling2D)

# from multilabel_CM import *

from keras.models import load_model

from sklearn.metrics import accuracy_score

from keras.metrics import binary_accuracy



import tensorflow as tf

from sklearn.model_selection import StratifiedShuffleSplit

from keras import regularizers

#from sklearn.metrics import confusion_matrix

############################



#### !!!!!!!!!!! This is path to the dataset !!!!!!!!!!!!!!!!!!!!

#### must be changed to actual path to the dataset

print(os.listdir("../input/C-NMC_training_data"))

# new shapes for i,age resizing



fftn = 150

fftm = 150



# prepare data frame with path to images and labels - 'all' or 1 for cancer, 'hem' or 0 for health



def prepare(path):

    image_list = []

    label = []

    path_list = [path + 'fold_0', path + 'fold_1', path + 'fold_2']

    for i in path_list:

        dirr_all = i + '/'+ 'all'

        dirr_hem = i + '/'+ 'hem'

        listdir_all = os.listdir(dirr_all)

        listdir_hem = os.listdir(dirr_hem)

        for j in listdir_all:

            image_list.append(dirr_all+ '/' + j)

            label.append(1)

        for j in listdir_hem:

            image_list.append(dirr_hem+ '/' + j)

            label.append(0)

    df = pd.DataFrame(list(zip(image_list, label)), columns =['Img', 'label']) 

    return df



# resize image for speed up calculations 



def preproc_resize(img):

#    new_img =  color.rgb2gray(img)

    new_img = img

    new_img = resize(img, (fftn,fftm,3),anti_aliasing=True)

    

    return new_img



# splitting to traing and test datasets       



def train_test_split(test_ratio, df):

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

    for train_index, test_index in split.split(df,df["label"]):

        strat_train = df.loc[train_index]

        strat_test = df.loc[test_index]

    return strat_train, strat_test 

    

# getting images and separate to xTrain, yTrain, xTest, yTest



def train_test_image(train, test):

    

    xTrain = np.empty(shape=(len(train), fftn, fftm,3))

    xTest = np.empty(shape=(len(test), fftn, fftm,3))

    yTrain = np.empty(shape=(len(train), 2))

    yTest = np.empty(shape=(len(test), 2))



    train_count = 0

    test_count = 0

    

    for i in list(train.index):

        print(train['Img'][i])

        img = preproc_resize(io.imread(train['Img'][i]))

#        img = io.imread(train['Img'][i])

        xTrain[train_count,] = img

        if train['label'][i] == 1:

            yTrain[train_count,] = [0,1]

        if train['label'][i] == 0:

            yTrain[train_count,] = [1,0]



        train_count += 1

    for i in list(test.index):

        print(test['Img'][i])

        img = preproc_resize(io.imread(test['Img'][i]))

#        img = io.imread(test['Img'][i])

        xTest[test_count,] = img

        if test['label'][i] == 1:

            yTest[test_count,] = [0,1]

        if test['label'][i] == 0:

            yTest[test_count,] = [1,0]

        test_count += 1





    return xTrain, yTrain, xTest, yTest  

    

# neural net model

    

def model(xTrain, yTrain, xTest, yTest):

    kernel_initializer='lecun_uniform'

    bias_initializer='lecun_uniform'

    kernel_regularizer=None

    activation = "selu"

    

    batch_size = 8

    data_rows = fftn

    data_cols = fftm

    input_shape = (data_rows, data_cols,3)

    nb_epoch = 20 # learning epochs

    alpha_zero = 0.001 # learning rate

    

    model = models.Sequential()

    model.add(Conv2D(32, (3, 3), input_shape = input_shape, border_mode="same",

                      data_format="channels_last",kernel_initializer=kernel_initializer, 

                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(3,3)))



    model.add(Conv2D(64, (3, 3), input_shape = input_shape, border_mode="same",

                      data_format="channels_last",kernel_initializer=kernel_initializer, 

                      bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(3,3)))

    

    model.add(Conv2D(128, (3, 3), input_shape = input_shape, border_mode="same",

                  data_format="channels_last",kernel_initializer=kernel_initializer, 

                  bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

    model.add(Activation(activation))

    model.add(MaxPooling2D(pool_size=(3,3)))



    model.add(Conv2D(256, (3, 3), input_shape = input_shape, border_mode="same",

                  data_format="channels_last",kernel_initializer=kernel_initializer, 

                  bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer))

    model.add(Activation(activation))

    model.add(Dropout(0.8))

    model.add(MaxPooling2D(pool_size=(3,3)))





   

     ############ adding fully connected layers

    model.add(Flatten())

    model.add(Dense(256,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))

    model.add(Activation(activation))

    model.add(Dropout(0.8))

    model.add(Dense(128,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))

    model.add(Activation(activation))

    model.add(Dropout(0.8))

    model.add(Dense(64,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))

    model.add(Activation(activation))

    model.add(Dropout(0.6))

    model.add(Dense(32,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer))





    model.add(Dense(2))

    model.add(Activation('sigmoid'))

    model.summary()



    optimizer1 = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, 

                                  schedule_decay=0.004)

    model.compile(loss=["binary_crossentropy"],optimizer=optimizer1, metrics=["accuracy"])

    history = model.fit(xTrain, yTrain, batch_size = batch_size, epochs = nb_epoch, 

                        verbose=2, validation_data = (xTest, yTest))

    return model, history

    

# evaluating the accuracy of the model



def evaluate(model, xTest, yTest):

    score = model.evaluate(xTest, yTest,verbose = 0)

    print("test score: %f" % score[0])

    print("test accuracy: %f" % score[1])

    yPred = model.predict(xTest)

    CM = multilabel_confusion_matrix(yTest, np.round_(yPred))

    print("----------confusion_matrix---------\n",CM)

    print(model.summary())      



# visualise accuracy and loss plots

    

def visualise(history):

    print(history.history.keys())

    # summarize history for accuracy

    plt.plot(history.history['acc'], 'k-')

    plt.plot(history.history['val_acc'], 'k:')

    plt.title('Model accuracy')

    

    plt.ylabel('accuracy')

    plt.xlabel('epochs')

    plt.legend(['train', 'test'], loc='center right')

    plt.show()

    # summarize history for loss

    plt.plot(history.history['loss'], 'k-')

    plt.plot(history.history['val_loss'], 'k:')

    plt.title('Model loss')

    plt.ylabel('loss')

    plt.xlabel('epochs')

    plt.legend(['train', 'test'], loc='center right')

    plt.show()

        

# predict class for new image    

    

def pred_image(model, image):

    img = io.imread(image)

    new_img = resize(img, (fftn,fftm,3),anti_aliasing=True)

    x = np.empty(shape=(1, fftn, fftm,3))

    x[0,] = new_img

    yPred = model.predict(x)

    return print('yPred', yPred)    



df = prepare('../input/C-NMC_training_data/')

df.groupby(['label']).count()

train, test = train_test_split(0.4, df)

train.groupby(['label']).count()

test.groupby(['label']).count()



xTrain, yTrain, xTest, yTest = train_test_image(train, test)



list(train.index)



# a = xTrain[1]



m,hist = model(xTrain, yTrain, xTest, yTest)

evaluate(m, xTest, yTest)

visualise(hist)

# all, must be [0,1]

# pred_image(m,'D:/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data/1.bmp')

# normal, must be [1,0]

# pred_image(m,'D:/C-NMC_test_prelim_phase_data/C-NMC_test_prelim_phase_data/4.bmp')
