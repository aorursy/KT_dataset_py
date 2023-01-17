import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train.head()
train.describe().loc['max',].max()
train.shape, test.shape
Y_train = train['label'].values

X_train = (train.loc[:, 'pixel0':] / 255).values



X_train.shape, Y_train.shape
X_test = (test / 255).values
from keras import models, optimizers

from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras.layers import Dense, Dropout

import tensorflow as tf

import tensorflow_addons as tfa

np.random.seed(42)

tf.random.set_seed(42)



model = models.Sequential()

model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer=optimizers.Adam(lr=1e-3), 

              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])



def lr_scheduler(epoch, lr):

    return lr * 0.9



checkpoint_path = 'bestmodel2.hdf5'

checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_sparse_categorical_accuracy', 

                             verbose=0, save_best_only=True, mode='max')



scheduler = LearningRateScheduler(lr_scheduler, verbose=0)



early_stop = EarlyStopping(monitor='var_loss', min_delta=0, patience=5, mode='min', verbose=0)



tqdm_callback = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False, 

                                              leave_overall_progress=True, 

                                              show_epoch_progress=False,

                                              show_overall_progress=True)



callbacks_list = [checkpoint, scheduler, tqdm_callback, early_stop]





history = model.fit(X_train, Y_train, batch_size=200, epochs=100, 

                    callbacks=callbacks_list, verbose=0, validation_split=0.2)
def graph_plot(history):

    

    for i in history.history.keys():

        print(f'{i}\nmin = {min(history.history[i])}, max = {max(history.history[i])}\n')

    

    epoch = len(history.history['loss'])

    for k in list(history.history.keys()):

        if 'val' not in k:

            plt.figure(figsize=(10, 7))

            plt.plot(history.history[k])

            if k != 'lr':

                plt.plot(history.history['val_' + k])

            plt.title(k, fontsize=10)



            plt.ylabel(k)

            plt.xlabel('epoch')

            plt.grid()



            plt.yticks(fontsize=10, rotation=30)

            plt.xticks(fontsize=10, rotation=30)

            plt.legend(['train', 'test'], loc='upper left', fontsize=10, title_fontsize=15)

            plt.show()

            

graph_plot(history)
from keras.layers import Conv2D, MaxPooling2D, Flatten



X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))



model2 = models.Sequential()

model2.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))



model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(128, (3, 3), activation='relu'))



model2.add(MaxPooling2D((2, 2)))

model2.add(Conv2D(96, (3, 3), activation='relu'))



model2.add(Flatten())

model2.add(Dense(128, activation='relu'))

model2.add(Dropout(0.6))

model2.add(Dense(64, activation='relu'))

model2.add(Dropout(0.6))

model2.add(Dense(10, activation='softmax'))



model2.compile(optimizer=optimizers.Adam(lr=1e-3), 

              loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])



checkpoint_path2 = 'bestmodel2.hdf5'

checkpoint2 = ModelCheckpoint(checkpoint_path2, monitor='val_sparse_categorical_accuracy', 

                             verbose=0, save_best_only=True, mode='max')



scheduler2 = LearningRateScheduler(lr_scheduler, verbose=0)



early_stop2 = EarlyStopping(monitor='var_loss', min_delta=0, patience=5, mode='min', verbose=0)



tqdm_callback2 = tfa.callbacks.TQDMProgressBar(leave_epoch_progress=False, 

                                              leave_overall_progress=True, 

                                              show_epoch_progress=False,

                                              show_overall_progress=True)



callbacks_list2 = [checkpoint2, scheduler2, tqdm_callback2, early_stop2]





history2 = model2.fit(X_train, Y_train, batch_size=390, epochs=30, 

                    callbacks=callbacks_list2, verbose=0, validation_split=0.2)
graph_plot(history2)
model2.load_weights(checkpoint_path2)

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
submit = pd.DataFrame(np.argmax(model2.predict(X_test), axis=1), columns=['Label'], 

                      index=pd.read_csv('../input/digit-recognizer/sample_submission.csv')['ImageId'])



submit.index.name = 'ImageId'

submit.to_csv('submittion.csv')
submit