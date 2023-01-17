from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
import time

import pandas as pd

import numpy as np



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, BatchNormalization

from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import model_from_json
# random seed

seed = 33

np.random.RandomState(seed)



# validation to training split ration

valid_size = 0.1



# use data augmentation i nthe first part of training

to_augment = False
data_path = '../input/Kannada-MNIST/'



train_path = data_path + 'train.csv'

test_path = data_path + 'test.csv'

dig_path = data_path + 'Dig-MNIST.csv'

sample_path = data_path + 'sample_submission.csv'



save_path = ''

load_path = '../input/kennada-mnist-pretrained-model/'
train_df = pd.read_csv(train_path)

test_df = pd.read_csv(test_path)

dig_df = pd.read_csv(dig_path)

sample_df = pd.read_csv(sample_path)
# convert dataframes to numpy matricies

X = train_df.drop('label', axis=1).to_numpy()

y = train_df['label'].to_numpy()

X_dig = dig_df.drop('label', axis=1).to_numpy()

y_dig = dig_df['label'].to_numpy()

X_test = test_df.drop('id', axis=1).to_numpy()



# reshape X's for keras and encode y using one-hot-vector-encoding

X = X.reshape(-1, 28, 28, 1)

y = to_categorical(y)

X_dig = X_dig.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)



# normalize the data to range(0, 1)

X = X / 255

X_dig = X_dig / 255

X_test = X_test / 255



# split to train and validation sets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=valid_size, random_state=seed) 



print('X_train shape = {}'.format(X_train.shape))

print('Y_train shape = {}'.format(y_train.shape))

print('X_valid shape = {}'.format(X_valid.shape))

print('Y_valid shape = {}'.format(y_valid.shape))
# model builder

def build_model(optimizer):

    model = Sequential()

    

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(256, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
# save model

def save_trained_model(model, save_path, optimizer):

    # serialize model to JSON

    model_json = model.to_json()

    with open('{}Kennada MNIST with {}.json'.format(save_path, optimizer), "w") as json_file:

        json_file.write(model_json)



    # serialize weights to HDF5

    model.save_weights('{}Kennada MNIST with {}.h5'.format(save_path, optimizer))



    

# load pretrained model

def load_trained_model(optimizers, optimizer, load_path):

    # load json and create model

    json_file = open('{}Kennada MNIST with {}.json'.format(load_path, optimizers[optimizer]), 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    model = model_from_json(loaded_model_json)



    # load weights into new model

    model.load_weights('{}Kennada MNIST with {}.h5'.format(load_path, optimizers[optimizer]))

    

    # compile the model

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    

    return model
def load_history(load_path, optimizer):

    history = pd.read_csv('{}Kennada MNIST with {}.csv'.format(load_path, optimizer))

    

    return history.to_dict('list')



def save_history(history, save_path, optimizer):

    hist_df = pd.DataFrame(history)

    hist_df.to_csv('{}Kennada MNIST with {}.csv'.format(save_path, optimizer), index=False)
# integer or None. Number of samples per gradient update. If unspecified, batch_size will default to 32

batch_size = 1024

# integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch

verbose = 0

# integer. Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided

epochs = 30
# every optimizer has a name

optimizers = {

    'sgd':        'SGD',

    'rmsprop':    'RMSprop',

    'adagrad':    'Adagrad',

    'adadelta':   'Adadelta',

    'adam':       'Adam',

    'adamax':     'Adamax',

    'nadam':      'Nadam',

}



# and default learning rate

learning_rates = {

    'sgd':        1e-2,

    'rmsprop':    1e-3,

    'adagrad':    1e-2,

    'adadelta':   1.0,

    'adam':       1e-3,

    'adamax':     2e-3,

    'nadam':      2e-3,

}
# create learning rate decay callback borrowed from here: https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                            patience=3, 

                                            verbose=0, 

                                            factor=0.5, 

                                            min_lr=0.00001)



# artificially increase training set

train_datagen = ImageDataGenerator(rescale=1.0,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)



# artificially increase validation set

valid_datagen = ImageDataGenerator(rescale=1.0)
# prepare empty dictionaries

history = {}

model = {}



for n, optimizer in enumerate(optimizers):

    # build model for every optimizer

    model[optimizer] = build_model(optimizer)



    # measure training time

    start = time.time()



    # train model

    if to_augment:

        h = model[optimizer].fit_generator(train_datagen.flow(X_train, y_train, batch_size=batch_size),

                                           steps_per_epoch=100,

                                           epochs=epochs,

                                           validation_data=valid_datagen.flow(X_valid, y_valid),

                                           callbacks=[learning_rate_reduction],

                                           verbose=verbose)

    else:

        h = model[optimizer].fit(X_train,

                                 y_train,

                                 batch_size=batch_size,

                                 epochs=epochs,

                                 validation_data=(X_valid,y_valid),

                                 callbacks=[learning_rate_reduction],

                                 verbose=verbose)



    history[optimizer] = h.history



    # print results

    print("{0} Optimizer: ".format(optimizers[optimizer]))

    print("Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}, Training time={3:.2f} minutes"

              .format(epochs, 

                      max(history[optimizer]['accuracy']), 

                      max(history[optimizer]['val_accuracy']), 

                      (time.time()-start)/60))
# apply smoothing filter

def smoothing_filter(data, filter_n=3):

    # filter_n should be odd number

    # extend the end for better accuracy at the end

    data = np.concatenate((data, [data[-1]]*filter_n))

    

    # apply filter

    data = np.convolve(data, [1/filter_n]*filter_n)

    

    # remove filter delay and padding

    return data[int(np.ceil(filter_n/2)) : -filter_n]





# plot training accuracy helper function

def plot_training_accuracy(history, names, epochs, to_smooth=False, filter_n=3, styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']):

    # filter_n should be odd number

    plt.figure(figsize=(15, 5))

    

    for n, h in enumerate(history.values()):

        # get validation accuracy history

        val_acc = h['val_accuracy']

        

        # smooth on request

        if to_smooth:

            val_acc = smoothing_filter(val_acc, filter_n)

        

        # plot history

        plt.plot(val_acc, linestyle=styles[n])

    

    plt.title('Model validation accuracy')

    plt.ylabel('accuracy')

    plt.xlabel('epoch')

    plt.legend(names, loc='upper left')

    axes = plt.gca()

    axes.set_ylim([0.99, 0.997])

    axes.set_xlim([0, epochs-1])
# plot learning hystory for all optimizers

plot_training_accuracy(history, optimizers.values(), epochs)