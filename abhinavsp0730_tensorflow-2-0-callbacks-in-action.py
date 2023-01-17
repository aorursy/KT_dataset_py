# installing the gdown package

! conda install -y gdown
import gdown



!gdown  https://drive.google.com/uc?id=1-4cbmjy9LmUiR8-V3905oZquU0mIGuOZ

!gdown  https://drive.google.com/uc?id=1-5ZP_djvEH9-WGp1jdk-KBAi8yMnbQjt

!gdown  https://drive.google.com/uc?id=1-7jMARlD5EkPWiKgXsqeN8b76NlHD2EL



!gdown  https://drive.google.com/uc?id=1-Dp8LamSxKhBGrC1ObWq9zwBMb2jKxXI

!gdown  https://drive.google.com/uc?id=1-TDB9Adm8PDLUcHh2dvu9bdy8wf081N6

!gdown  https://drive.google.com/uc?id=1-WN5SCMeyftsxHqgeqp1Bf8ReTUo6mEG
import tensorflow as tf

import tensorflow_datasets as tfds

import numpy as np



images_train = np.load("./images_train.npy") / 255

images_valid = np.load("./images_valid.npy") / 255

images_test = np.load("./images_test.npy") / 255

labels_train = np.load("./labels_train.npy")

labels_valid = np.load("./labels_valid.npy")

labels_test= np.load("./labels_test.npy")



print("{} training data examples".format(images_train.shape[0]))

print("{} validation data examples".format(images_valid.shape[0]))

print("{} test data examples".format(images_test.shape[0]))
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D , MaxPool2D

def get_compiled_model(compile=True):

  ''' prepare and compile the model '''

  model = Sequential()

  model.add(Conv2D(32, (3,3), activation='relu', padding='SAME', input_shape=(160,160,3)))

  model.add(Conv2D(32, (3,3), activation='relu', padding='SAME'))

  model.add(MaxPool2D(2,2))

  model.add(Conv2D(64, (3,3), activation='relu', padding='SAME'))

  model.add(Conv2D(64, (3,3), activation='relu', padding='SAME'))

  model.add(MaxPool2D(2,2))

  model.add(Flatten())

  model.add(Dense(128, activation='relu'))

  model.add(Dense(1, activation='sigmoid'))

  

  if compile is True:

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),

              loss=tf.keras.losses.BinaryCrossentropy(), 

              metrics=[tf.keras.metrics.BinaryAccuracy(name='acc')])

  return model



model = get_compiled_model()



# inspecting the model architecture

model.summary()
import datetime

from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):

  def on_train_begin(self,logs=None):

    print("Training is started, at time {}".format(datetime.datetime.now().time()))

  def on_train_end(self, logs=None):

    print("Training is ended at {}".format(datetime.datetime.now().time()))

  def on_train_batch_begin(self, batch, logs=None):

    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):

    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))



custom_callback = CustomCallback()



model = get_compiled_model()

model.fit(images_train, labels_train, validation_data=(images_valid, labels_valid), 

          epochs=1, callbacks=[custom_callback])
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', 

                               min_delta=0.001, 

                               patience=2, 

                               verbose=0, 

                               mode='min', 

                               baseline=None, 

                               restore_best_weights=False)



model = get_compiled_model()

model.fit(images_train, labels_train, 

          validation_data=(images_valid, labels_valid), 

          epochs=80, 

          callbacks=[early_stopping])
from tensorflow.keras.callbacks import ReduceLROnPlateau

callback  = ReduceLROnPlateau(monitor='val_loss', 

                              factor=0.1, 

                              patience=10, 

                              verbose=0, 

                              mode='auto', 

                              min_delta=0.002, 

                              cooldown=0, 

                              min_lr=0)



model = get_compiled_model()

model.fit(images_train, labels_train, 

          validation_data=(images_valid, labels_valid), 

          epochs=20, 

          callbacks=[callback])
from tensorflow.keras.callbacks import ModelCheckpoint

model_checkpoint_callback = ModelCheckpoint(filepath= "model.h5", 

                                            monitor='val_loss', 

                                            verbose=0, 

                                            save_best_only=False, 

                                            save_weights_only=False, 

                                            mode='min', 

                                            save_freq='epoch')

model = get_compiled_model()

model.fit(images_train, labels_train, 

          validation_data=(images_valid, labels_valid), 

          epochs=20, 

          callbacks=[model_checkpoint_callback])



# loading the model from the disk.

model.load_weights("model.h5")
# inspecting the architecture .

model.summary()
# evaluating the model on the test set.

model.evaluate(images_test, labels_test)
from tensorflow.keras.callbacks import LearningRateScheduler





def lr_function(epoch, lr):

    if epoch % 2 == 0:

        return lr

    else:

        return lr + epoch/1000



learning_rate_schedular_callback = LearningRateScheduler(schedule= lr_function ,

                                                         verbose=1)



model = get_compiled_model()



model.fit(images_train, labels_train, 

          validation_data=(images_valid, labels_valid), 

          epochs=10, 

          callbacks=[learning_rate_schedular_callback] )

# making a custom callback for the 

class CustomCallback(tf.keras.callbacks.Callback):



    def __init__(self, metrics_dict, num_epochs='?', log_frequency=1,

                 metric_string_template='\033[1m[[name]]\033[0m = \033[94m{[[value]]:5.3f}\033[0m'):

        super().__init__()



        self.metrics_dict = collections.OrderedDict(metrics_dict)

        self.num_epochs = num_epochs

        self.log_frequency = log_frequency



        log_string_template = 'Epoch {0:2}/{1}: '

        separator = '; '



        i = 2

        for metric_name in self.metrics_dict:

            templ = metric_string_template.replace('[[name]]', metric_name).replace('[[value]]', str(i))

            log_string_template += templ + separator

            i += 1



        log_string_template = log_string_template[:-len(separator)]

        self.log_string_template = log_string_template



    def on_train_begin(self, logs=None):

        print("Training: \033[92mstart\033[0m.")



    def on_train_end(self, logs=None):

        print("Training: \033[91mend\033[0m.")



    def on_epoch_end(self, epoch, logs={}):

        if (epoch - 1) % self.log_frequency == 0 or epoch == self.num_epochs:

            values = [logs[self.metrics_dict[metric_name]] for metric_name in self.metrics_dict]

            print(self.log_string_template.format(epoch, self.num_epochs, *values))

import collections

import functools



 

metrics_to_print = collections.OrderedDict([("loss", "loss"), 

                                             ("acc", "acc"),

                                            ("v-loss", "val_loss"),

                                            ("v-acc", "val_acc"),

                                            ])





custom_callback = CustomCallback(metrics_to_print, num_epochs=5)
model.fit(images_train, labels_train, validation_data=(images_valid, labels_valid), epochs=5, 

          callbacks=[custom_callback], verbose=False )