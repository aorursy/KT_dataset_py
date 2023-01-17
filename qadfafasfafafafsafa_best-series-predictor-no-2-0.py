# Магическое слово для отрисовки их в юпитер ноутбуке

%matplotlib inline



# Для повторяемости результатов

RANDOM_SEED = 42



# Для нейронок

import keras

import numpy as np

import pandas as pd

import matplotlib as plot

import matplotlib.pyplot as plt

import tensorflow as tf

import random

import torch

import torchvision

from sklearn.model_selection import * 

from catboost import *

from keras import *

from keras.layers.convolutional import Conv2D

from keras.layers import *

from tensorflow.nn import *

from keras.callbacks import *

from keras.models import *

from keras.optimizers import *

from keras.preprocessing import image

from sklearn.metrics import accuracy_score

from datetime import datetime

from sklearn.metrics import log_loss, roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler



# Фиксируем рэндом сид для повторяемости результатов

r_np = np.random.seed(RANDOM_SEED)

r_tf = tf.set_random_seed(RANDOM_SEED)



np.random.seed(RANDOM_SEED)

tf.set_random_seed(RANDOM_SEED)
# X = pd.read_csv('../input/train_df.csv', index_col = 'id').drop(['y'], axis = 1)

# y = pd.read_csv('../input/train_df.csv', index_col = 'id').y.to_frame()

# X_test = pd.read_csv('../input/test_df.csv', index_col = 'id')
# from catboost import *



# clf = CatBoostClassifier(n_estimators = 12000, verbose = 0, eval_metric = 'AUC', task_type = 'GPU')

# clf.fit(X, y) # , use_best_model = True, eval_set = (X_val, y_val), plot = True)



# y_pred = clf.predict(X_test)

# sub = pd.DataFrame({'y': y_pred, 'id': X_test.index})

# sub.to_csv('submission.csv', index = False)
# class roc_auc_callback(Callback):

#     def __init__(self,training_data,validation_data):

#         self.x = training_data[0]

#         self.y = training_data[1]

#         self.x_val = validation_data[0]

#         self.y_val = validation_data[1]



#     def on_train_begin(self, logs={}):

#         return



#     def on_train_end(self, logs={}):

#         return



#     def on_epoch_begin(self, epoch, logs={}):

#         return



#     def on_epoch_end(self, epoch, logs={}):

#         y_pred = self.model.predict(self.x, verbose=0)

#         roc = roc_auc_score(self.y, y_pred)

#         logs['roc_auc'] = roc_auc_score(self.y, y_pred)

#         logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1



#         y_pred_val = self.model.predict(self.x_val, verbose=0)

#         roc_val = roc_auc_score(self.y_val, y_pred_val)

#         logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)

#         logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1



#         print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')

#         return



#     def on_batch_begin(self, batch, logs={}):

#         return



#     def on_batch_end(self, batch, logs={}):

#         return

    

# def timer(start_time=None):

#     if not start_time:

#         start_time = datetime.now()

#         return start_time

#     elif start_time:

#         thour, temp_sec = divmod(

#             (datetime.now() - start_time).total_seconds(), 3600)

#         tmin, tsec = divmod(temp_sec, 60)

#         print('\n Time taken: %i hours %i minutes and %s seconds.' %

#               (thour, tmin, round(tsec, 2)))
# skf = StratifiedKFold(n_splits = folds, random_state = 512)

# starttime = timer(None)



# for i, (train_index, test_index) in enumerate(skf.split(X, y)):

#     start_time = timer(None)

#     X_train, X_val = X.iloc[train_index], X.iloc[test_index]

#     y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    
# import keras

# import tensorflow

# from keras.models import *

# from keras.layers import *



# net = Sequential()



# # net.add(Dense(512, activation = relu))

# # net.add(Dense(128, activation = relu)) 

# # net.add(Dropout(rate = 0.9))

# # net.add(Dense(1, activation = sigmoid)) 



# features = X_train.shape[1]

# input_dim = 1



# net.add(Conv1D(3, 3, activation = 'relu', input_shape = (features, input_dim))) 

# net.add(MaxPool1D(pool_size = 2))

# net.add(Flatten())

# net.add(Dense(512, activation = 'relu'))

# net.add(Dense(1, activation = 'sigmoid'))



# net.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.001), metrics = ['accuracy'])
# skf = StratifiedKFold(n_splits = folds, random_state = 512)

# starttime = timer(None)



# for i, (train_index, test_index) in enumerate(skf.split(X, y)):

#     start_time = timer(None)
from __future__ import print_function

import keras

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

import os



batch_size = 16

num_classes = 10

epochs = 100

patience = 15

data_augmentation = False

num_predictions = 20

save_dir = os.path.join(os.getcwd(), 'saved_models')

model_name = 'keras_cifar10_trained_model.h5'



# The data, split between train and test sets:

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print(x_train.shape[0], 'train samples')

print(x_test.shape[0], 'test samples')



# Convert class vectors to binary class matrices.

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(32, (2, 2), padding = 'valid', input_shape = x_train.shape[1:]))

model.add(Activation('relu'))

model.add(Conv2D(32, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.15))



model.add(Conv2D(64, (2, 2), padding = 'same'))

model.add(Activation('relu'))

model.add(Conv2D(64, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(512))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes))

model.add(Activation('softmax'))



# initiate optimizers

# opt = keras.optimizers.rmsprop(lr = 0.0001, decay = 1e-6)

# adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.001)



# Let's train the model using RMSprop

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])



x_train = x_train.astype('float32')

x_test = x_test.astype('float32')

x_train /= 255

x_test /= 255



callbacks = [ReduceLROnPlateau(monitor = 'val_acc', factor = 0.15, patience = 7, verbose = 2, mode = 'auto'), ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 2, factor = 0.5, min_lr = 0.00001), EarlyStopping(monitor = 'val_acc', patience = patience, mode = 'max', verbose = 0), ModelCheckpoint('best_model.h5', monitor = 'val_acc', mode = 'max', save_best_only = True, verbose = 0)]  



if not data_augmentation:

    print('Not using data augmentation.')

    model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (x_test, y_test), shuffle = True, callbacks = callbacks, verbose = 2)

else:

    print('Using real-time data augmentation.')

    # This will do preprocessing and realtime data augmentation:

    datagen = ImageDataGenerator(

        featurewise_center = False,  # set input mean to 0 over the dataset

        samplewise_center = False,  # set each sample mean to 0

        featurewise_std_normalization = True,  # divide inputs by std of the dataset

        samplewise_std_normalization = False,  # divide each input by its std

        zca_whitening = False,  # apply ZCA whitening

        zca_epsilon = 1e-06,  # epsilon for ZCA whitening

        rotation_range = 0,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)

        shear_range = 0.15,  # set range for random shear

        zoom_range = 0.2,  # set range for random zoom

        channel_shift_range = 0.,  # set range for random channel shifts

        #   set mode for filling points outside the input boundaries

        fill_mode = 'nearest',

        cval = 0.,  # value used for fill_mode = "constant"

        horizontal_flip = True,  # randomly flip images

        vertical_flip = True,  # randomly flip images

        #   set rescaling factor (applied before any other transformation)

        rescale = 0.2,

        #   set function that will be applied on each input

        preprocessing_function = None,

        #   image data format, either "channels_first" or "channels_last"

        data_format = None,

        #   fraction of images reserved for validation (strictly between 0 and 1)

        validation_split = 0.0)

    

    # Compute quantities required for feature-wise normalization

    # (std, mean, and principal components if ZCA whitening is applied).

    datagen.fit(x_train)

    

    # Fit the model on the batches generated by datagen.flow().

    model = model.fit_generator(datagen.flow(x_train, y_train, batch_size = batch_size), 

                        steps_per_epoch = x_train.shape[0] * 2, epochs = epochs, validation_data = (x_test, y_test), workers = -1, verbose = 1, callbacks = callbacks)

    

    del model

    model = load_model('best_model.h5')



# Save model and weights

if not os.path.isdir(save_dir):

    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)

model.save(model_path)

print('Saved trained model at %s ' % model_path)



# Score trained model.

scores = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', scores[0])

print('Test accuracy:', scores[1])
res = pd.DataFrame({'loss': scores[0], 'score': scores[1]}, index = list('0')) 

res.to_csv('result.csv') 

res.head() 
# for run in range(runs):

#         print('\n Fold %d - Run %d\n' % ((i + 1), (run + 1)))

#         np.random.seed()



# callbacks = [roc_auc_callback(training_data = (X_train, y_train), validation_data = (X_val, y_val)), EarlyStopping(monitor = 'val_loss', patience = patience, mode = 'max', verbose = 1), CSVLogger('keras-5fold-run-01-v1-epochs.log', separator = ',', append = False), ModelCheckpoint('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check', monitor = 'val_loss', mode = 'max', save_best_only = True,verbose = 1)]  



# fit = net.fit(X_train, y_train, batch_size = batchsize, epochs = 500, verbose = 0, validation_data = (X_val, y_val), callbacks = callbacks) 
# del net



# net = load_model('keras-5fold-run-01-v1-fold-' + str('%02d' % (i + 1)) + '-run-' + str('%02d' % (run + 1)) + '.check')

# scores_val_run = net.predict(X_val, verbose=0)

# LL_run = log_loss(y_val, scores_val_run)

# print('\n Fold %d Run %d Log-loss: %.5f' % ((i + 1), (run + 1), LL_run))

# AUC_run = roc_auc_score(y_val, scores_val_run)

# print(' Fold %d Run %d AUC: %.5f' % ((i + 1), (run + 1), AUC_run))

# print(' Fold %d Run %d normalized gini: %.5f' % ((i + 1), (run + 1), AUC_run*2 - 1))

# y_pred_run = net.predict(X_test, verbose = 0)



# if run > 0:

#     scores_val = scores_val_run + scores_val_run

#     y_pred = y_pred_run + y_pred_run

# else:

#     scores_val = scores_val_run

#     y_pred = y_pred_run

    

# scores_val = scores_val / runs

# y_pred = y_pred / runs

# LL = log_loss(y_val, scores_val)

# print('\n Fold %d Log-loss: %.5f' % ((i + 1), LL))

# AUC = roc_auc_score(y_val, scores_val)

# print(' Fold %d AUC: %.5f' % ((i + 1), AUC))

# print(' Fold %d normalized gini: %.5f' % ((i + 1), AUC*2-1))

# timer(start_time)



# # if i > 0:

# #     fpred = y_pred + y_pred

# #     avreal = np.concatenate((avreal, y_val), axis=0)

# #     avpred = np.concatenate((avpred, scores_val), axis=0)

# #     avids = np.concatenate((avids, val_ids), axis=0)



# fpred = y_pred

# avreal = y_val

# avpred = scores_val

# avids = y_val.index

    

# pred = fpred

# cv_LL = cv_LL + LL

# cv_AUC = cv_AUC + AUC

# cv_gini = cv_gini + (AUC*2 - 1)
# LL_oof = log_loss(avreal, avpred)

# print('\n Average Log-loss: %.5f' % (cv_LL/folds))

# print(' Out-of-fold Log-loss: %.5f' % LL_oof)

# AUC_oof = roc_auc_score(avreal, avpred)

# print('\n Average AUC: %.5f' % (cv_AUC/folds))

# print(' Out-of-fold AUC: %.5f' % AUC_oof)

# print('\n Average normalized gini: %.5f' % (cv_gini/folds))

# print(' Out-of-fold normalized gini: %.5f' % (AUC_oof*2-1))

# score = str(round((AUC_oof*2-1), 5))

# timer(starttime)

# mpred = pred / folds
# print('#\n Writing results')

# now = datetime.now()

# oof_result = pd.DataFrame(avreal, columns=['y'])

# oof_result['y'] = avpred

# oof_result['id'] = avids

# oof_result.sort_values('id', ascending=True, inplace=True)

# oof_result = oof_result.set_index('id')

# sub_file = 'train_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'

# print('\n Writing out-of-fold file:  %s' % sub_file)

# oof_result.to_csv(sub_file, index=True, index_label='id')
# result = pd.DataFrame(mpred, columns=['y'])

# result['id'] = X_test.index

# result = result.set_index('id')

# print('\n First 10 lines of your 5-fold average prediction:\n')

# print(result.head(10))

# sub_file = 'submission_' + str(score) + '_' + str(now.strftime('%Y-%m-%d-%H-%M')) + '.csv'

# print('\n Writing submission:  %s' % sub_file)

# result.to_csv(sub_file, index=True, index_label='id')