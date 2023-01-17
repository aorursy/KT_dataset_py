import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import keras.models as md

import keras.layers.core as core

import keras.utils.np_utils as kutils

import keras.layers.convolutional as conv



from keras.layers import MaxPool2D



"""

    Created on 29/01/2018

    @author: Dimitri Belli

    Version: 1.03



    Train and test set: MNIST dataset

    Tensor manipulation library: TensorFlow (GPU)

    Core: Convolutional Neural Networks with Keras

    Structure: Simple script



    The code has been rearranged in order to define a more precise structure and effective execution

    Both train and test set are apart from keras models and you must set them in the same directory

    of the script

    

    Multiple GPUs speed up the computation and refine the prediction percentage

    

    As suggested in the notes within the code, modifying hyperparameters and epoch it is possible to 

    obtain accuracy percentages between 99,17% and 99,39% 

"""



# variables

train_filename = "train.csv"

test_filename = "test.csv"



# dimensions for reshaping images whose values are arranged in vectors (tuple is also possible)

pict_rows, pict_cols = 28, 28



# alternatively: 64 and 128

filters_1 = 32

filters_2 = 64



# dimensions 

convol = 3



# set to 100 to obtain a higher value of accuracy

epoch = 50



# Maintain this value. The lower the batch size value, the less accuracy will be

batch_size = 128



def test_the_model(file, rows, cols, cnn):

    test  = pd.read_csv(file).values



    # reshape images from a vector to a 3D matrix normalizing the gray-scale

    test_images = test.reshape(test.shape[0], rows, cols, 1)

    test_images = test_images.astype(float)



    test_images /= 255.0



    predictions = cnn.predict_classes(test_images)



    np.savetxt('MNIST_predictions.csv', np.c_[range(1,len(predictions)+1),predictions], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')



def train_and_test_the_model(train_file, test_file, rows, cols, fil_1, fil_2, cnv, epc, batch):

    train = pd.read_csv(train_file).values



    # reshape images from a vector to a 3D matrix normalizing the grayscale

    train_images = train[:, 1:].reshape(train.shape[0], rows, cols, 1)

    train_images = train_images.astype(float)

    train_images /= 255.0



    train_labels = kutils.to_categorical(train[:, 0])

    nb_classes = train_labels.shape[1]



    # building the convolutional neural network with keras

    cnn = md.Sequential()



    cnn.add(conv.Conv2D(fil_1, kernel_size = (cnv,cnv),  padding='same', activation="relu", input_shape=(rows, cols, 1)))

    cnn.add(MaxPool2D(strides=(2,2))) # downsampling



    cnn.add(conv.Conv2D(fil_2, kernel_size = (cnv,cnv), padding='same', activation="relu"))

    cnn.add(MaxPool2D(strides=(2,2))) # downsampling



    cnn.add(core.Flatten())

    cnn.add(core.Dropout(0.2))

    cnn.add(core.Dense(128, activation="relu"))

    cnn.add(core.Dense(nb_classes, activation="softmax"))



    cnn.summary()



    # Using adamax as optimizer is possible to increase the accuracy percentage 

    cnn.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])



    # best results (without overfitting) have been obtained with a number of epoch set to 100 using a 3 GPUs machine

    cnn.fit(train_images, train_labels, batch_size=batch, nb_epoch=epc, verbose=1)

    

    # it calls the previous definition

    test_the_model(test_file, rows, cols, cnn)



# Execution

train_and_test_the_model(train_filename, test_filename, pict_rows, pict_cols, filters_1, filters_2, convol, epoch, batch_size)
