# kaggle paths

TRAIN_DATA_DIR = "../input/10-monkey-species/training/training/"

VALID_DATA_DIR = "../input/10-monkey-species/validation/validation/"

LABELS_FILE = "../input/10-monkey-species/monkey_labels.txt"

FILE_SAVE_PATH = "/kaggle/working/"
import os

import numpy as np

# set the seed for to make results reproducable

np.random.seed(2)

import pandas as pd





import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('darkgrid')





from tensorflow.random import set_seed

# set the seed for to make results reproducable

set_seed(2)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
img_width = 224  # since this is the size of image, that vgg19 model, as the documentation states, accepts

img_height = 224 # check: https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19#arguments_1

btch_size = 16   # since the dataset is small
# Since the number of example is very low, I am making some augmentation in the training data(not increasing the count of training examples),

# to make the robust to the unseen examples.

train_data_IDG = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,

                                    zoom_range=0.2, fill_mode='nearest', horizontal_flip=True, rescale=1.0/255,

                                    validation_split=0.0)



valid_data_IDG = ImageDataGenerator(rescale=1.0/255)
train_data_generator = train_data_IDG.flow_from_directory(directory=TRAIN_DATA_DIR, target_size=(img_width, img_height),

                                                          class_mode='categorical', batch_size=btch_size)



valid_data_generator = valid_data_IDG.flow_from_directory(directory=VALID_DATA_DIR, target_size=(img_width, img_height),

                                                          class_mode='categorical')
train_len = 1_098

valid_len = 272
tr_imgs, tr_labels = train_data_generator.next()

tr_imgs.shape, tr_labels.shape
target_data = pd.read_csv(filepath_or_buffer=LABELS_FILE)

target_data
i = 2

print(target_data.iloc[np.where(tr_labels[i] == 1)[0][0], 2])

plt.imshow(tr_imgs[i])
i = 7

print(target_data.iloc[np.where(tr_labels[i] == 1)[0][0], 2])

plt.imshow(tr_imgs[i])
labels = train_data_generator.classes

labels.shape
sns.distplot(a=labels, bins=None, kde=False)
pd.Series(labels).value_counts(normalize=True)
from tensorflow.keras.applications.vgg19 import VGG19





from tensorflow.keras import Sequential, Model

from tensorflow.keras import layers
base_model = VGG19(include_top=False, weights='imagenet', input_shape=(224,224,3), pooling='max')





# # let's make some layers at the last layers non-trainable, since that's where I want to make changes at

for layer in base_model.layers[:-6]:

    layer.trainable = False



# checking the status in number of parameters trainable

base_model.summary()
for layer in base_model.layers[:]:

    print(layer, layer.trainable)
model_vgg19 = Sequential()



for layer in base_model.layers:

    model_vgg19.add(layer)



model_vgg19.add(layers.Dense(512, activation="relu"))

model_vgg19.add(layers.Dropout(0.5))

model_vgg19.add(layers.Dense(10, activation="softmax"))



model_vgg19.summary()
import time

from tensorflow.keras.callbacks import Callback



class EpochTimeHistory(Callback):

    """

    a custom callback to print the time(in minutes, to console) each epoch took during.

    """

    def on_train_begin(self, logs={}):

        self.train_epoch_times = []

        self.valid_epoch_times = []



    def on_epoch_begin(self, epoch, logs={}):

        self.epoch_time_start = time.time()



    def on_epoch_end(self, epoch, logs={}):

        cur_epoch_time = round((time.time() - self.epoch_time_start)/60, 4)

        self.train_epoch_times.append(cur_epoch_time)

        print(" ;epoch {0} took {1} minutes.".format(epoch+1, cur_epoch_time))





    def on_test_begin(self, logs={}):

        self.test_time_start = time.time()



    def on_test_end(self, logs={}):

        cur_test_time = round((time.time() - self.test_time_start)/60, 4)

        self.valid_epoch_times.append(cur_test_time)

        print(" ;validation took {} minutes.".format(cur_test_time))
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



# saving model weights at the end of every epoch, in case something happens to my internet connection; and so i can resume training with those weights from the epoch last session stopped at.

model_save_cb = ModelCheckpoint(filepath=os.path.join(FILE_SAVE_PATH, 'vgg19-weights-epoch{epoch:02d}-val_fbeta_score{val_fbeta_score:.2f}.h5'))



# let's not waste resources for performance that we aren't gonna get

early_stop_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min')



# also reducing learning rate, because you know all about the gradient reaching optima in correct way.

reduce_learning_rate_cb = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, cooldown=2, min_lr=0.00001, verbose=1)



# to see how much time each epoch took to complete training

epoch_times_cb = EpochTimeHistory()
from tensorflow_addons.metrics import FBetaScore

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import CategoricalAccuracy, Precision, Recall

# from tensorflow.keras.losses import CategoricalCrossentropy





model_vgg19.compile(loss = 'categorical_crossentropy',

                optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001),

                metrics = [FBetaScore(num_classes=10, average='macro', name='fbeta_score'),

                           CategoricalAccuracy(name='cat_acc'),

                           Precision(name='precision'), Recall(name='recall')])
history_VGG19 = model_vgg19.fit(train_data_generator,

                                  steps_per_epoch=train_len // btch_size,

                                  validation_data=valid_data_generator,

                                  validation_steps=valid_len // btch_size,

                                  epochs=35, verbose=1,

                                  callbacks=[model_save_cb, early_stop_cb,

                                             epoch_times_cb, reduce_learning_rate_cb])
model_vgg19.save(filepath=FILE_SAVE_PATH, overwrite=True, include_optimizer=True)
# if training had no problems, delete all the saved weights and include the final model

!rm /kaggle/working/*.h5
plt.plot(history_VGG19.history['loss'])

plt.plot(history_VGG19.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='center right')

plt.show()
plt.plot(history_VGG19.history['fbeta_score'])

plt.plot(history_VGG19.history['val_fbeta_score'])

plt.title('model fbeta_score')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='center right')

plt.show()
plt.plot(history_VGG19.history['cat_acc'])

plt.plot(history_VGG19.history['val_cat_acc'])

plt.title('model categorical accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='center right')

plt.show()
plt.plot(history_VGG19.history['precision'])

plt.plot(history_VGG19.history['recall'])

plt.plot(history_VGG19.history['val_precision'])

plt.plot(history_VGG19.history['val_recall'])

plt.title('model precision and recall')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['precision_train', 'recall_train', 'precision_val', 'recall_val'], loc='center right')

plt.show()