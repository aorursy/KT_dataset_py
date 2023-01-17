# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pathlib import Path



%matplotlib inline

import matplotlib.pyplot as plt



import tensorflow as tf

from tensorflow import keras

import numpy as np

import pandas as pd



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, Dropout, Dense, Flatten, BatchNormalization, MaxPooling2D

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import LearningRateScheduler

from tensorflow.keras.applications.resnet50 import ResNet50



from sklearn.model_selection import train_test_split
lr_schedule = LearningRateScheduler(lambda epoch: 1e-6 * 2**(epoch/2))
train_path = Path.cwd()/'..'/'input'/'Kannada-MNIST'/'train.csv'



all_data = pd.read_csv(train_path)
y = all_data.label

x = all_data.iloc[:,1:].values

x = x.reshape(x.shape[0], 28, 28, 1)
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=10000,

                                                     random_state=42, shuffle=True,

                                                     stratify=y)
train_datagen = ImageDataGenerator(rescale=1./255.,

                                   rotation_range=10,

                                   width_shift_range=0.25,

                                   height_shift_range=0.25,

                                   shear_range=0.1,

                                   zoom_range=0.25,

                                   horizontal_flip=False)



valid_datagen = ImageDataGenerator(rescale=1./255.)



train_generator = train_datagen.flow(x_train, y_train, batch_size=1024)

valid_generator = valid_datagen.flow(x_valid, y_valid)
def plot_history(history):

    #-----------------------------------------------------------

    # Retrieve a list of list results on training and test data

    # sets for each training epoch

    #-----------------------------------------------------------

    acc      = history.history[     'accuracy' ]

    val_acc  = history.history[ 'val_accuracy' ]

    loss     = history.history[    'loss' ]

    val_loss = history.history['val_loss' ]



    epochs   = range(len(acc)) # Get number of epochs



    #------------------------------------------------

    # Plot training and validation accuracy per epoch

    #------------------------------------------------

    plt.plot  ( epochs,     acc , label='Training')

    plt.plot  ( epochs, val_acc , label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Accuracy')

    plt.legend ()

    plt.title ('Training and validation accuracy')

    plt.figure()



    #------------------------------------------------

    # Plot training and validation loss per epoch

    #------------------------------------------------

    plt.plot  ( epochs,     loss, label='Training')

    plt.plot  ( epochs, val_loss, label='Validation')

    plt.xlabel ('Epoch')

    plt.ylabel ('Loss')

    plt.legend ()

    plt.title ('Training and validation loss'   )
def report(model, history=None, validation_generator=None):

    if history is not None:

        plot_history(history)

    

    if validation_generator is not None:

        # Evaluate trained model on validation set

        validation_generator.reset()

        [val_loss, val_acc] = model.evaluate_generator(validation_generator)

        print('Model evaluation')

        print(f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

        print()
def plot_learning_rate_history(history, steps, rate_sched, axisrange):

    lrs = [rate_sched(n) for n in range(steps)]

    plt.semilogx(lrs, history.history["loss"])

    plt.axis(axisrange)

    plt.xlabel('Learning rate')

    plt.ylabel('Loss')

    plt.show()
from tensorflow.keras.callbacks import Callback

from tensorflow.keras import backend as K



class LR_Updater(Callback):

    '''This callback is utilized to log learning rates every iteration (batch cycle)

    it is not meant to be directly used as a callback but extended by other callbacks

    ie. LR_Cycle

    '''

    def __init__(self, epoch_iterations):

        '''

        iterations = training batches

        epoch_iterations = number of batches in one full training cycle

        '''

        self.epoch_iterations = epoch_iterations

        self.trn_iterations = 0.

        self.history = {}

    def on_train_begin(self, logs={}):

        self.trn_iterations = 0.

        logs = logs or {}

    def on_batch_end(self, batch, logs=None):

        logs = logs or {}

        self.trn_iterations += 1

        K.set_value(self.model.optimizer.lr, self.setRate())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)



class LR_Cycle(LR_Updater):

    '''This callback is utilized to implement cyclical learning rates

    it is based on this pytorch implementation https://github.com/fastai/fastai/blob/master/fastai

    and adopted from this keras implementation https://github.com/bckenstler/CLR

    '''

    

    def __init__(self, iterations, cycle_mult = 1):

        '''

        iterations = initial number of iterations in one annealing cycle

        cycle_mult = used to increase the cycle length cycle_mult times after every cycle

        for example: cycle_mult = 2 doubles the length of the cycle at the end of each cy$

        '''

        self.min_lr = 0

        self.cycle_mult = cycle_mult

        self.cycle_iterations = 0.

        super().__init__(iterations)

    

    def setRate(self):

        self.cycle_iterations += 1

        if self.cycle_iterations == self.epoch_iterations:

            self.cycle_iterations = 0.

            self.epoch_iterations *= self.cycle_mult

        decay_phase = np.pi*self.cycle_iterations/self.epoch_iterations

        decay = (np.cos(decay_phase) + 1.) / 2.

        return self.max_lr * decay

    

    def on_train_begin(self, logs={}):

        super().on_train_begin(logs={})

        self.cycle_iterations = 0.

        self.max_lr = K.get_value(self.model.optimizer.lr)
def build_model(optimizer=Adam()):

    model = Sequential()



    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(64, kernel_size=5, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(128, kernel_size=5, padding='same', activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(256, kernel_size=3, activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Flatten())

    model.add(Dense(256))

    model.add(BatchNormalization())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Dense(10, activation='softmax'))

    

    model.compile(loss='sparse_categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])

    

    return model
model = build_model(Adam(learning_rate=1e-3))



model.summary()
cyclic_lr = LR_Cycle(49, cycle_mult=2)
history = model.fit_generator(train_generator,

                              epochs=31,

                              validation_data=valid_generator,

                                callbacks=[cyclic_lr])
report(model, history, valid_generator)
test_path = Path.cwd()/'..'/'input'/'Kannada-MNIST'/'test.csv'



all_data_test = pd.read_csv(test_path)
x_test = all_data_test.iloc[:,1:].values

x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
predictions = model.predict_classes(x_test/255.)
output = pd.DataFrame({'id': all_data_test.id,

                       'label': predictions})

output.to_csv("submission.csv",index=False)