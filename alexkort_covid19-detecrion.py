import numpy as np

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, load_img

import random

from os import listdir

from os.path import isfile, join

import keras.callbacks as kcall

import tensorflow as tf
# TPU detection  

# try:

#   tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# except ValueError:

#   tpu = None



# # TPUStrategy for distributed training

# if tpu:

#   tf.config.experimental_connect_to_cluster(tpu)

#   tf.tpu.experimental.initialize_tpu_system(tpu)

#   strategy = tf.distribute.experimental.TPUStrategy(tpu)

# else: # default strategy that works on CPU and single GPU

#   strategy = tf.distribute.get_strategy()



train_dir = '../input/xray-covid19/train/'

test_dir = '../input/xray-covid19/test/'

labels = ['pneumonia', 'COVID-19', 'normal']

# img_width, img_height, channels = 500, 500, 3

color_mode = 'rgb'

batch_size = 8

epochs = 10

datagen = ImageDataGenerator(

#                     rescale=1./255    

                    samplewise_center=True,

                    samplewise_std_normalization=True

                    )

train_generator = datagen.flow_from_directory(train_dir, target_size = (500, 500), batch_size = batch_size, color_mode = color_mode, class_mode='categorical')

test_generator = datagen.flow_from_directory(test_dir, target_size = (500, 500), batch_size = batch_size, color_mode = color_mode, class_mode='categorical')



class_weight={0:15.6,1:1.,2:1.46}



train_size = 13897

test_size = 300
from keras.metrics import Precision, Recall, Accuracy

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_f1s = []

        self.val_recalls = []

        self.val_precisions = []

    

    def on_epoch_end(self, epoch, logs={}):

        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()

        val_targ = self.model.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict)

        _val_recall = recall_score(val_targ, val_predict)

        _val_precision = precision_score(val_targ, val_predict)

        self.val_f1s.append(_val_f1)

        self.val_recalls.append(_val_recall)

        self.val_precisions.append(_val_precision)

        print(f" — val_f1: {_val_f1} — val_precision: {_val_precision} — val_recall {_val_recall}")

        return

from keras.applications import InceptionV3, ResNet50V2, DenseNet201

from keras.optimizers import Adam



inception = InceptionV3(

    include_top=True,

    weights=None,

    input_shape=(500,500,3),

    classes=3

)



resNet = ResNet50V2(

    include_top=True,

    weights=None,

    input_shape=(500,500,3),

    classes=3

)



denseNet = DenseNet201(

    include_top=True,

    weights=None,

    input_shape=(500,500,3),

    classes=3

)

metrics = ['accuracy']

# inception.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[Metrics()])

# resNet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[Metrics()])

# denseNet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[Metrics()])

inception.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=metrics)

resNet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=metrics)

denseNet.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=metrics)



inception_checkpoint = ModelCheckpoint('inception_adam.hdf5', monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

inception_log = CSVLogger('inception_training_adam.log')



inception.fit(train_generator,

        steps_per_epoch = train_size // batch_size,

        epochs = epochs,

        callbacks = [inception_checkpoint, inception_log],      

        class_weight = class_weight,

        validation_data = test_generator)

inception.save('inception')

resNet_checkpoint = ModelCheckpoint('resNet_adam.hdf5', monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

resNet_log = CSVLogger('resNet_training_adam.log')



resNet.fit(train_generator,     

        steps_per_epoch = train_size // batch_size,

        epochs = epochs,

        callbacks = [resNet_checkpoint, resNet_log],

        class_weight = class_weight,

        validation_data = test_generator)

resNet.save('resNet')

denseNet_checkpoint = ModelCheckpoint('denseNet_adam.hdf5', monitor = "val_loss", verbose = 1,

                              save_best_only = True, mode = "min")

denseNet_log = CSVLogger('densnet_training_adam.log')

denseNet.fit(train_generator,

        steps_per_epoch = train_size // batch_size,

        epochs = epochs,

        callbacks = [denseNet_checkpoint, denseNet_log],

        class_weight = class_weight,

        validation_data = test_generator)

denseNet.save('denseNet')
target_names = ['COVID-19', 'normal', 'pneumonia']



Y_pred = inception.predict_generator(test_generator, test_size // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)

print('Inception')

print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')

print(classification_report(test_generator.classes, y_pred, target_names=target_names))



Y_pred = resNet.predict_generator(test_generator, test_size // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)

print('ResNet')

print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')

print(classification_report(test_generator.classes, y_pred, target_names=target_names))





Y_pred = denseNet.predict_generator(test_generator, test_size // batch_size+1)

y_pred = np.argmax(Y_pred, axis=1)

print('DenseNet')

print('Confusion Matrix')

print(confusion_matrix(test_generator.classes, y_pred))

print('Classification Report')

print(classification_report(test_generator.classes, y_pred, target_names=target_names))