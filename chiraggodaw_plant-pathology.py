# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

from __future__ import absolute_import, division, print_function, unicode_literals

# from tensorflow.keras.applications.inception_v3 import ResNet101V2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc

import cv2

import sklearn.metrics

import itertools

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from tensorflow import keras

from matplotlib import pyplot as plt

from keras.utils import to_categorical

import tensorflow_hub as hub





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
gc.collect()

IMAGE_HEIGHT = 224 #350

IMAGE_WIDTH = 224 #234

train_df = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

# train_df = train_df.sample(frac = 1)

train_df  = train_df.sort_values(by = 'multiple_diseases', ascending = False)

col  = train_df['multiple_diseases'] == 1

df_try = train_df[col]

train_df = train_df.append([df_try]*1,ignore_index=True)



train_x_images = np.array(train_df['image_id'])

train_y = []

for i,j in train_df.iterrows():

    train_y.append(list(j)[1:])

train_y = np.array(train_y)

file_name = np.array(train_df['image_id'])

train_X = []

gc.collect()

for i in file_name:

    image = (cv2.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/" + i + ".jpg"))

    resized = cv2.resize(image, (IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_AREA)

    train_X.append(resized- [40,40,40])







print(train_df)
gc.collect()



train_X = np.array(train_X)

training_y = []

for i in train_y:

    training_y.append((np.where(i==1)[0][0]))

training_y = np.array(training_y)    

    

gc.collect()



training_X = train_X[0:1216]

train_y = training_y [0:1216]

test_X = train_X[1216:1821]

test_Y = training_y[1216:1821]



gc.collect()



train_datagen = ImageDataGenerator(

#     rescale=1. / 255,

    rotation_range=360,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.3,

    shear_range = 0.1,

    horizontal_flip=True,

    vertical_flip = True,

    fill_mode='nearest')





model = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2, 2),

   tf.keras.applications.Xception( weights = 'imagenet',include_top = False,input_shape=(IMAGE_WIDTH,IMAGE_HEIGHT,3)),

#     efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=(300,300,3)),

#         feature_extractor,

#     tf.keras.layers.Conv2D(128, (3, 3),activation='relu'),

#     tf.keras.layers.MaxPooling2D(3, 3),

#     tf.keras.layers.Flatten(),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dropout(0.5),

 

    tf.keras.layers.Dense(512,activation = tf.nn.relu),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512,activation = tf.nn.relu,kernel_regularizer = tf.keras.regularizers.l2()),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(4,activation = tf.nn.softmax)

])



model.compile(optimizer = tf.keras.optimizers.Adamax(),

          loss = 'categorical_crossentropy',

#             loss = tf.keras.losses.CategoricalHinge(),

          metrics=['accuracy'])









#--------------------------------------------------------------------------------#

# train_datagen = ImageDataGenerator(

#     rotation_range=360,

#     width_shift_range=0.2,

#     height_shift_range=0.2,

#     zoom_range=0.2,

#     horizontal_flip=True,

#     fill_mode='nearest')



# model = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(170,256,3)),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(128, activation = 'relu'),

#     tf.keras.layers.Dense(4,activation = tf.nn.softmax)

# ])



# model.compile(optimizer = tf.keras.optimizers.RMSprop(),

#           loss = 'categorical_crossentropy',

#           metrics=['accuracy'])

#---------------------------------------------------------------------------------#
class_names = ['healthy', 'multiple_diseases','rust','scab']
y_binary = to_categorical(training_y)

y_binary_train = to_categorical(train_y)

y_binary_test = to_categorical(test_Y)

def plot_confusion_matrix(cm, class_names):

    """

    Returns a matplotlib figure containing the plotted confusion matrix.

    

    Args:

       cm (array, shape = [n, n]): a confusion matrix of integer classes

       class_names (array, shape = [n]): String names of the integer classes

    """

    

    figure = plt.figure(figsize=(8, 8))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title("Confusion matrix")

    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks, class_names, rotation=45)

    plt.yticks(tick_marks, class_names)

    

    # Normalize the confusion matrix.

    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    

    # Use white text if squares are dark; otherwise black.

    threshold = cm.max() / 2.

    

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        color = "white" if cm[i, j] > threshold else "black"

        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    return figure
gc.collect()

annealer = ReduceLROnPlateau(monitor='accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-5)

checkpoint = ModelCheckpoint('model.h5', verbose=1, save_best_only=True)



# class myCallback(tf.keras.callbacks.Callback):

#     def on_epoch_end(self, epoch, logs={}):

#         if((logs.get('accuracy')>0.999 and logs.get('val_loss')<0.20) or(logs.get('accuracy')>0.999 and logs.get('val_accuracy')>0.97) ):

#             print("\nReached 99.9% accuracy so cancelling training!")

#             self.model.stop_training = True
gc.collect()

# callbacks = myCallback()



history = model.fit_generator(train_datagen.flow(training_X, y_binary_train, batch_size= 32 ),

                              steps_per_epoch=len(training_X) / 32,

                              validation_data = (test_X, y_binary_test),

                              epochs=70,

                              shuffle = True,

                              callbacks=[annealer],)
gc.collect()
# model = tf.keras.models.load_model('model.h5')

# model.evaluate(test_X, y_binary_test)
gc.collect()



test_pred_raw = model.predict(test_X)

y_binary_test_raw = np.argmax(y_binary_test,axis = 1)



test_pred = np.argmax(test_pred_raw, axis=1)



cm = sklearn.metrics.confusion_matrix(y_binary_test_raw, test_pred)



figure = plot_confusion_matrix(cm, class_names=class_names)
model.evaluate(test_X, y_binary_test)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Val accuracy')

plt.title('Training accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Training Loss')

plt.title('Training Loss')

plt.legend(loc=0)

plt.figure()





plt.show()
gc.collect()

test = []

for i in range(0,1821):

    image = (cv2.imread("/kaggle/input/plant-pathology-2020-fgvc7/images/Test_" + str(i) + ".jpg"))

    resized = cv2.resize(image, (IMAGE_HEIGHT,IMAGE_WIDTH), interpolation = cv2.INTER_AREA)

    test.append(resized - [40,40,40])
gc.collect()

results = []

for i in test:

    result = model.predict(np.array([i]))

    results.append(result[0])


df = pd.DataFrame(results, columns = ['healthy', 'multiple_diseases','rust','scab']) 

image_id = []

for  i in range(0,1821):

    image_id.append("Test_"+str(i))

df.insert(0,"image_id",image_id,False)
df.to_csv('submission.csv', index=False)
df