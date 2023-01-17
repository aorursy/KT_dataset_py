import tensorflow as tf

from keras import backend as K

from keras.utils import Sequence, to_categorical

from keras.applications.densenet import DenseNet121

from keras.callbacks import TensorBoard, EarlyStopping

from keras.optimizers import Adam

from keras import layers, Model, metrics

from skimage.io import imread, imshow

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import types

import os
#Sequence generator, based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class CactusSequence(Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=128, dim=(32,32,3),

                 n_classes=2, shuffle=True, is_test=False, 

                 data_dir="data/train"):

        """

        Initialize sequence generator

        @param list_IDs: array of filenames to load

        @param labels: array or dict of labels corresponding to items in list_IDs. 

                       In case of list we suppose that labels are aligned with IDs.

                       Ignored if is_test=True.

        @param batch_size: size of the batch. Last batch can be smaller that this value.

        @param dim: dimensions of the single sample (i.e. 32x32 RGB image should have dimensions (32,32,3)

        @param n_classes: number of classes for classification

                          Ignored if is_test=True.

        @param shuffle: should we shuffle our data before producing the next batch

        @param is_test: should we yield label for each sample (i.e. test sequence doesn't have label)

        """

        self.dim = dim

        self.batch_size = batch_size

        self.list_IDs = list_IDs

        self.indexes = np.arange(len(self.list_IDs))

        self.is_test = is_test

        self.shuffle = shuffle

        self.data_dir = data_dir

        

        if not is_test:    

            self.n_classes = n_classes    

            self.labels = labels

            if not isinstance(labels, dict):

                self.labels = dict(zip(list_IDs, labels))

        

            self.on_epoch_end()        



    def __len__(self):

        'Denotes the number of batches per epoch'

        l = int(np.floor(len(self.list_IDs) / self.batch_size))+1

        return l



    def __getitem__(self, index):

        'Generate one batch of data'

        # Generate indexes of the batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]



        # Find list of IDs

        list_IDs_temp = [self.list_IDs[k] for k in indexes]



        # Generate data

        # Note that we don't include label for test sequence

        if not self.is_test:

            X, y = self.__data_generation_train(list_IDs_temp)

            return X, y

        else:

            X = self.__data_generation_test(list_IDs_temp)

            return X



    def on_epoch_end(self):

        'Updates indexes after each epoch'

        if self.shuffle == True and not self.is_test:

            np.random.shuffle(self.indexes)



    def __data_generation_train(self, list_IDs_temp):

        'Generates data containing not more than batch_size samples with labels' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((len(list_IDs_temp), *self.dim))

        y = np.empty((len(list_IDs_temp)), dtype=np.int32)



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            X[i,] = imread(self.data_dir+"/"+ID) / 255



            # Store class

            y[i] = self.labels[ID]

            

        return X, to_categorical(y, num_classes=self.n_classes)

    

    def __data_generation_test(self, list_IDs_temp):

        'Generates data containing not more than batch_size samples without labels' # X : (n_samples, *dim, n_channels)

        # Initialization

        X = np.empty((len(list_IDs_temp), *self.dim))



        # Generate data

        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            X[i,] = imread(self.data_dir+"/"+ID) / 255

            

        return X
#AUC-score function

def auc(y_true, y_pred):

    auc = tf.py_func(lambda y_true, y_pred : roc_auc_score( y_true, y_pred, average='macro', sample_weight=None).astype('float32'),

                     [y_true, y_pred],

                     'float32',

                     stateful=False,

                     name='sklearnAUC')

    return auc
train_data = pd.read_csv("../input/train.csv", dtype={"id": str, "has_cactus": np.int32})

train_data = train_data.sample(frac=1).reset_index(drop=True)

print("Total samples in train set:", len(train_data))
#note that our classes are disbalanced

train_data.describe()
test_data = os.listdir("../input/test/test")

print("Total samples in test set:", len(test_data))
#we'll divide train set into train and validation parts 

train_data, val_data = train_data[:12000], train_data[12000:]
#Instantiating our generators

train_generator = CactusSequence(train_data.id.values, train_data.has_cactus.values, data_dir="../input/train/train")

val_generator = CactusSequence(val_data.id.values, val_data.has_cactus.values, data_dir="../input/train/train", shuffle=False)

test_generator = CactusSequence(test_data, is_test=True, data_dir="../input/test/test")
#We'll use Densenet architecture and train it from scratch. We exclude last year so we use our own output shape.

densenet_model = DenseNet121(include_top=False, input_shape=(32,32,3), weights=None)
#We flatten the last layer of cutted densenet model and use sigmoid activation to produce probabilities for 2 classes.

x = layers.Flatten()(densenet_model.output)

predictions = layers.Dense(2, activation="sigmoid")(x)
#final model

model = Model(inputs=densenet_model.input, outputs=predictions)
#using binary_crossentropy as a loss function and binary_accuracy as a metric (along with ROC AUC on validation)

model.compile(optimizer=Adam(lr=0.000005), loss='binary_crossentropy', metrics=[metrics.binary_accuracy, auc])
#adding earlystop callback. Optionaly adding tensorboard callback.

earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=10, restore_best_weights=True)
#fitting our model. Note that we set class_weight according to disbalance factor.

model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=50, callbacks=[earlystop_callback], class_weight={0: 3.0, 1:1.0})
y_pred = model.predict_generator(test_generator)

result = pd.DataFrame(data={"id":test_data, "has_cactus":y_pred[:,1]})

result.to_csv("densenet_30epochs.csv", index=False)