# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

import matplotlib.pyplot as plt

from PIL import Image



from keras.optimizers import Adam, RMSprop, SGD

from keras.models import Sequential, Model

from keras.layers import Dropout, LeakyReLU, Conv2D, Conv1D, Flatten

from keras.layers import Dense, Activation, BatchNormalization

from keras.callbacks import EarlyStopping



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
## init variable

dataset_directory = "../input"

train_path = "{0}/{1}".format(dataset_directory, "train.csv")

test_path = "{0}/{1}".format(dataset_directory, "test.csv")

sample_path = "{0}/{1}".format(dataset_directory, "sample_submission.csv")

x_train_pickle_path = "{0}/{1}".format(dataset_directory, "x_train_data.npy")

y_train_pickle_path = "{0}/{1}".format(dataset_directory, "y_train_data.npy")

x_test_pickle_path = "{0}/{1}".format(dataset_directory, "x_test_data.npy")



def show_train_history(train_history, train, validation):

    plt.plot(train_history.history[train])

    plt.plot(train_history.history[validation])

    plt.title('Train History')

    plt.ylabel(train)

    plt.xlabel('Epoch')

    plt.legend(['train', 'validation'], loc='upper left')

    plt.show()



def load_bin_data():

    print ("[*] Read train_x data to {0}".format(x_train_pickle_path))

    train_x = np.load(x_train_pickle_path)

    print ("[*] Read train_y data to {0}".format(y_train_pickle_path))

    train_y = np.load(y_train_pickle_path)

    print ("[*] Read test_x data to {0}".format(x_test_pickle_path))

    test_x = np.load(x_test_pickle_path)



    print("[+] Load data sucessfully.\n")

    return train_x, train_y, test_x



def load_data(save_pickle = True):

    print ("[*] Read csv data from {0} for training data".format(train_path))

    train_x = pd.read_csv(train_path)

    train_x = train_x.drop(['ID_code'], axis=1)

    train_x = train_x.drop(['target'], axis=1)

    train_x = train_x.values



    print ("[*] Read csv data from {0} for training label".format(train_path))

    train_y = pd.read_csv(train_path, usecols=['target'])

    train_y = train_y.values



    print ("[*] Read csv data from {0} from testing data".format(test_path))

    test_x = pd.read_csv(test_path)

    test_x = test_x.drop(['ID_code'], axis=1)



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

    print("[+] Start to write submission {0}.\n".format(submit_filename))

    read_header = pd.read_csv(sample_path, usecols=['ID_code'])

    result = pd.DataFrame({"ID_code": read_header.ID_code.values})

    result["target"] = y_pred

    result.to_csv(submit_filename, index=False)

    print("[+] Submission file has created {0}.\n".format(submit_filename))



def CNN_Model(x_train, y_train, x_val, y_val, x_test, lr=0.01, batch_size=200, epochs=100, model_filename="model_cnn.h5", submit_filename="submission_cnn.csv"):

    print ("-------------------------------------------------------------------------------------\n")

    print("[+] CNN Model.\n")



    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))

    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))



    model = Sequential()



    model.add(Conv1D(16, 2, input_shape=(x_train.shape[1], 1), activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv1D(16, 2, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv1D(16, 2, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.25))



    model.add(Dense(1, activation='sigmoid'))



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

                              callbacks=[earlystop],

                              verbose=2) 



    model.save(model_filename)

    y_pred = model.predict(x_test, batch_size=batch_size, verbose=2)

    _Submission(y_pred, submit_filename)



    return train_history
# training set

learning_rate = 0.0001

validation_split = 0.2

epochs = 200

batch_size = 1024

random_seed = 1234



x_train, y_train, x_test = load_data(False)

#x_train, y_train, x_test = load_bin_data()



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,

                                                  test_size=validation_split,

                                                  random_state=random_seed,

                                                  shuffle=True)



minmax_scale = MinMaxScaler(feature_range=(0, 1))

x_train = minmax_scale.fit_transform(x_train)

x_test = minmax_scale.fit_transform(x_test)

x_val = minmax_scale.fit_transform(x_val)
train_history_cnn = CNN_Model(x_train, y_train, x_val, y_val, x_test,

                              lr=learning_rate,

                              batch_size=batch_size,

                              epochs=epochs)
common.show_train_history(train_history_cnn, 'acc', 'val_acc')

common.show_train_history(train_history_cnn, 'loss', 'val_loss')