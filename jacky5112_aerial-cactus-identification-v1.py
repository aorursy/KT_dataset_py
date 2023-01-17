# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import math

import tarfile

import csv

import pandas as pd

import pickle as cPickle

import matplotlib.pyplot as plt

from PIL import Image



from keras.optimizers import Adam, RMSprop, SGD

from keras.models import Sequential, Model

from keras.layers import Dropout, LeakyReLU, Conv2D, MaxPooling2D, Flatten

from keras.layers import Dense, Activation, BatchNormalization, Embedding

from keras.layers import CuDNNLSTM

from keras.callbacks import EarlyStopping

from keras.applications.vgg16 import VGG16



from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/aerialcactusidentification"))

print(os.listdir("../input/aerial-cactus-identification"))



# Any results you write to the current directory are saved as output.
dataset_directory = "../input/aerialcactusidentification"

train_label = "{0}/{1}".format(dataset_directory, "train.csv")

#train_image_path = "{0}/{1}".format(dataset_directory, "train")

#test_image_path = "{0}/{1}".format(dataset_directory, "test")

train_path = "{0}/{1}".format(dataset_directory, "train.csv")

test_path = "{0}/{1}".format(dataset_directory, "test.csv")

sample_submission = "{0}/{1}".format("../input/aerial-cactus-identification", "sample_submission.csv")

x_train_pickle_path = "{0}/{1}".format(dataset_directory, "x_train_data.npy")

y_train_pickle_path = "{0}/{1}".format(dataset_directory, "y_train_data.npy")

x_test_pickle_path = "{0}/{1}".format(dataset_directory, "x_test_data.npy")



img_width, img_height, img_space = 32, 32, 3



def _build_header(training = True):

    if training == True:

        result = ["id", "label"]

    else:

        result = ["id"]



    for y in range(0, img_height):

        for x in range(0, img_width):

            result.append("{0}_{1}_R".format(y, x))

            result.append("{0}_{1}_G".format(y, x))

            result.append("{0}_{1}_B".format(y, x))



    return result



def _load_image( infilename ) :

    img = Image.open(infilename)

    img.load()

    img.thumbnail((img_width, img_height, 3), Image.ANTIALIAS)

    data = np.asarray(img, dtype="float32")

    data /= 255

    return data



def _pre_process_training():

    header = _build_header()

    y_train = pd.read_csv(train_label)

    y_train = y_train.values



    with open(train_path, "w") as write_train_data:

        writer = csv.writer(write_train_data)

        writer.writerow(header)



        for idx, val in enumerate(y_train):

            train_file_name = "{0}/{1}".format(train_image_path, val[0])

            row_data = [val[0], val[1]]



            if os.path.isfile(train_file_name) == True:

                img_data = _load_image(train_file_name)

                img_data = img_data.flatten()

                

                for _, img_val in enumerate(img_data):

                    row_data.append(img_val)



                writer.writerow(row_data)

                write_train_data.flush()

                

def _pre_process_testing():

    header = _build_header(False)

    y_test = pd.read_csv(sample_submission)

    y_test = y_test.values



    with open(test_path, "w") as write_test_data:

        writer = csv.writer(write_test_data)

        writer.writerow(header)



        for idx, val in enumerate(y_test):

            test_file_name = "{0}/{1}".format(test_image_path, val[0])

            row_data = [val[0]]



            if os.path.isfile(test_file_name) == True:

                img_data = _load_image(test_file_name)

                img_data = img_data.flatten()

                row_data.append(img_data)



                for _, img_val in enumerate(img_data):

                    row_data.append(img_val)



                writer.writerow(row_data)

                write_test_data.flush()

                

def load_bin_data():

    print ("[*] Read train_x data to {0}".format(x_train_pickle_path))

    train_x = np.load(x_train_pickle_path)

    print ("[*] Read train_y data to {0}".format(y_train_pickle_path))

    train_y = np.load(y_train_pickle_path)

    print ("[*] Read test_x data to {0}".format(x_test_pickle_path))

    test_x = np.load(x_test_pickle_path)



    print("[+] Load data sucessfully.\n")

    return train_x, train_y, test_x



def load_data(save_pickle=True, preprocess=False):

    if preprocess == True:

        print ("[*] Pre-process training data")

        _pre_process_training()

        print ("[*] Pre-process testing data")

        _pre_process_testing()



    print ("[*] Read csv data from {0} for training data".format(train_path))

    train_x = pd.read_csv(train_path)

    train_x = train_x.drop(['id'], axis=1)

    train_x = train_x.drop(['label'], axis=1)

    train_x = train_x.values



    print ("[*] Read csv data from {0} for training label".format(train_path))

    train_y = pd.read_csv(train_path, usecols=['label'])

    train_y = train_y.values



    print ("[*] Read csv data from {0} from testing data".format(test_path))

    test_x = pd.read_csv(test_path)

    test_x = test_x.drop(['id'], axis=1)

    test_x = test_x.values



    if save_pickle == True:

        print ("[*] Write train_x data to {0}".format(x_train_pickle_path))

        np.save(x_train_pickle_path, train_x)

        print ("[*] Write train_y data to {0}".format(y_train_pickle_path))

        np.save(y_train_pickle_path, train_y)

        print ("[*] Write test_x data to {0}".format(x_test_pickle_path))

        np.save(x_test_pickle_path, test_x)



        

    print("[+] Load data sucessfully.\n")

    return train_x, train_y, test_x



def _Submission(y_pred, submit_filename):

    ## submit

    read_header = pd.read_csv(sample_submission, usecols=['id'])

    result = pd.DataFrame({"id": read_header.id.values})

    result["has_cactus"] = y_pred

    result.to_csv(submit_filename, index=False)

    print("[+] Submission file has created {0}.\n".format(submit_filename))



def CNN_Model(x_train, y_train, x_val, y_val, x_test, lr=0.01, batch_size=20000, epochs=100, model_filename="model_cnn.h5", submit_filename="sample_submission.csv"):

    print ("-------------------------------------------------------------------------------------\n")

    print("[+] CNN Model.\n")



    x_train = np.reshape(x_train, (x_train.shape[0], img_width, img_height, img_space))

    x_val = np.reshape(x_val, (x_val.shape[0], img_width, img_height, img_space))

    x_test = np.reshape(x_test, (x_test.shape[0], img_width, img_height, img_space))



    model = Sequential()



    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_width, img_height, img_space), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='normal'))

    model.add(BatchNormalization())

    #model.add(Dropout(0.25))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu', kernel_initializer='normal'))

    #model.add(Dropout(0.25))



    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))



    model.compile(optimizer=Adam(lr=lr),

                  loss='binary_crossentropy',  

                  metrics=['accuracy'])



    print (model.summary())



    earlystop = EarlyStopping(monitor='val_loss', mode='auto', patience=10, verbose=1)



    train_history = model.fit(x=x_train, 

                          y=y_train,

                          validation_data=(x_val, y_val), 

                          epochs=epochs, 

                          batch_size=batch_size,

                          verbose=2,

                          callbacks=[earlystop]) 



    #model.save(model_filename)

    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)

    _Submission(y_pred, submit_filename)



    return train_history



def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()



# training set

learning_rate = 0.0001

validation_split = 0.2

epochs = 200

batch_size = 2048

random_seed = 12345

x_train, y_train, x_test = load_data(False, False)

#x_train, y_train, x_test = load_bin_data()



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,

                                                  test_size=validation_split,

                                                  random_state=random_seed)
train_history_cnn = CNN_Model(x_train, y_train, x_val, y_val, x_test,

                              lr=learning_rate,

                              batch_size=batch_size,

                              epochs=epochs,

                              submit_filename="sample_submission.csv"

                             )
show_train_history(train_history_cnn, 'acc', 'val_acc')

show_train_history(train_history_cnn, 'loss', 'val_loss')