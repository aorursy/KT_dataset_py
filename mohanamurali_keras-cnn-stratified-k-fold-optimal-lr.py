import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.models import model_from_json

from keras.layers import Conv2D, MaxPooling2D, MaxPool2D

from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization

from keras.optimizers import RMSprop, Adam

from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.optimizers import SGD

from keras.callbacks import EarlyStopping,ReduceLROnPlateau



from keras.layers import Dense

from keras.models import Model

import keras

import os

import glob

from skimage.io import imread

from keras.preprocessing import image





print(os.listdir("../input"))



train_dir = "../input/train/"

test_dir = "../input/test/"



dataset= pd.read_csv('../input/train_labels.csv',dtype='str')



def append_ext(fn):

    return fn+".tif"

dataset["id"]=dataset["id"].apply(append_ext)



datapath='../input/'

train_path = datapath+'train'

valid_path =  datapath+'train'

test_path=datapath+'test'
testing_files = glob.glob(os.path.join(test_dir,'*.tif'))

TESTING_BATCH_SIZE=10



submission = pd.DataFrame()

test_files = glob.glob(os.path.join(test_dir,'*.tif'))

submission = pd.DataFrame()

max_idx = len(test_files)

file_batch=5000

print('reading files')



test_df = pd.DataFrame({'path': test_files})

print(test_df.shape)

test_df['id'] = test_df.path.map(lambda x: x.split('/')[3].split(".")[0])

test_df['image'] = test_df['path'].map(imread)

test_df['label']=0

print('end of reading')

X_test = np.stack(test_df['image'].values)

X_test = keras.applications.resnet50.preprocess_input(X_test)
k_folds=1 
#Import Keras

import keras

from keras.models import Sequential

from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense

from keras.layers.normalization import BatchNormalization

from keras.layers.convolutional import SeparableConv2D

from keras.layers.core import Activation
# from https://www.kaggle.com/soumya044/histopathologic-cancer-detection

class CancerNet:

    @staticmethod

    def build(width, height, depth, classes):

        

        # initialize the model along with the input shape to be

        # "channels last" and the channels dimension itself

        model = Sequential()

        inputShape = (height, width, depth)

        chanDim = -1

        

        # CONV => RELU => POOL

        model.add(SeparableConv2D(32, (3, 3), padding="same",input_shape = inputShape))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # (CONV => RELU => POOL) * 2

        model.add(SeparableConv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(SeparableConv2D(64, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))



        # (CONV => RELU => POOL) * 3

        model.add(SeparableConv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(SeparableConv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(SeparableConv2D(128, (3, 3), padding="same"))

        model.add(Activation("relu"))

        model.add(BatchNormalization(axis=chanDim))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        

        # first (and only) set of FC => RELU layers

        model.add(Flatten())

        model.add(Dense(256))

        model.add(Activation("relu"))

        model.add(BatchNormalization())

        model.add(Dropout(0.2))



        # softmax classifier

        model.add(Dense(classes))

        model.add(Activation("sigmoid"))

        

        model.summary()



        # return the constructed network architecture

        return model
model= CancerNet.build(width = 96, height = 96, depth = 3, classes = 1)
# source :https://github.com/surmenok/keras_lr_finder

from matplotlib import pyplot as plt

import math

from keras.callbacks import LambdaCallback

import keras.backend as K





class LRFinder:

    """

    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.

    See for details:

    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0

    """

    def __init__(self, model):

        self.model = model

        self.losses = []

        self.lrs = []

        self.best_loss = 1e9



    def on_batch_end(self, batch, logs):

        # Log the learning rate

        lr = K.get_value(self.model.optimizer.lr)

        self.lrs.append(lr)



        # Log the loss

        loss = logs['loss']

        self.losses.append(loss)



        # Check whether the loss got too large or NaN

        if math.isnan(loss) or loss > self.best_loss * 4:

            self.model.stop_training = True

            return



        if loss < self.best_loss:

            self.best_loss = loss



        # Increase the learning rate for the next batch

        lr *= self.lr_mult

        K.set_value(self.model.optimizer.lr, lr)



    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1):

        num_batches = epochs * x_train.shape[0] / batch_size

        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))



        # Save weights into a file

        self.model.save_weights('tmp.h5')



        # Remember the original learning rate

        original_lr = K.get_value(self.model.optimizer.lr)



        # Set the initial learning rate

        K.set_value(self.model.optimizer.lr, start_lr)



        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))



        self.model.fit(x_train, y_train,

                        batch_size=batch_size, epochs=epochs,

                        callbacks=[callback])



        # Restore the weights to the state before model fitting

        self.model.load_weights('tmp.h5')



        # Restore the original learning rate

        K.set_value(self.model.optimizer.lr, original_lr)



    def plot_loss(self, n_skip_beginning=10, n_skip_end=5):

        """

        Plots the loss.

        Parameters:

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

        """

        plt.ylabel("loss")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])

        plt.xscale('log')



    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):

        """

        Plots rate of change of the loss function.

        Parameters:

            sma - number of batches for simple moving average to smooth out the curve.

            n_skip_beginning - number of batches to skip on the left.

            n_skip_end - number of batches to skip on the right.

            y_lim - limits for the y axis.

        """

        assert sma >= 1

        derivatives = [0] * sma

        for i in range(sma, len(self.lrs)):

            derivative = (self.losses[i] - self.losses[i - sma]) / sma

            derivatives.append(derivative)



        plt.ylabel("rate of loss change")

        plt.xlabel("learning rate (log scale)")

        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], derivatives[n_skip_beginning:-n_skip_end])

        plt.xscale('log')

        plt.ylim(y_lim)
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

for i in range(len(model.layers)):

    model.layers[i].trainable = True

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])



early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=2)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)

lr_finder = LRFinder(model)
#help function

def append_adress(fn):

    return "../input/train/"+fn 
#optimal LR will be search on 0.005 fraction of dataset

dataset_lr_finder=dataset.sample(frac=0.01)

print(dataset_lr_finder.columns)

dataset_lr_finder["path"]=dataset_lr_finder["id"].apply(append_adress) # adding image's path on optimal LR dataset

print(dataset_lr_finder.head(1))
dataset_lr_finder['image'] = dataset_lr_finder['path'].map(imread)

x_train = np.stack(dataset_lr_finder['image'].values)

x_train = keras.applications.resnet50.preprocess_input(x_train)

y_train =dataset_lr_finder.label
# Train a model with batch size 512 for 5 epochs

# with learning rate growing exponentially from 0.0001 to 1

lr_finder.find(x_train, y_train, start_lr=0.000001, end_lr=0.3, batch_size=100, epochs=10)
lr_finder.plot_loss()

#3.5.10-4
sgd = SGD(lr=3e-3, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
import keras

train_datagen=ImageDataGenerator(

 #   width_shift_range=0.5,

 #   height_shift_range=0.5,

 #   shear_range=0.5,

 #   zoom_range=0.5,

 #   horizontal_flip=True,vertical_flip=True,

 #   rotation_range=90,fill_mode = 'nearest',

    rescale=1/255,validation_split=0.3

    

  #  preprocessing_function= keras.applications.resnet50.preprocess_input

    )



train_generator = train_datagen.flow_from_dataframe(dataframe=dataset,directory=train_path,x_col = 'id',y_col = 'label',has_ext=False,

                subset='training',

                target_size=(96, 96),

                batch_size=32,

                class_mode='binary'

                )

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

    



validation_generator = train_datagen.flow_from_dataframe(

                dataframe=dataset,

                directory=valid_path,

                x_col = 'id',

                y_col = 'label',

                has_ext=False,

                subset='validation', # This is the trick to properly separate train and validation dataset

                target_size=(96, 96),

                batch_size=32,

                shuffle=False,

                class_mode='binary'

                )

STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

model.fit_generator(generator=train_generator,

              steps_per_epoch=STEP_SIZE_TRAIN,

              nb_epoch=10,

              shuffle=True,verbose=1,

              callbacks=[lr_reducer, early_stop],

              validation_data=validation_generator,

              validation_steps=STEP_SIZE_VALID)

predictions = model.predict(X_test)

    

test_df['label']+=predictions[:,0]
submission = test_df[["id", "label"]]


submission.to_csv("submission_14_03.csv", index = False, header = True)
