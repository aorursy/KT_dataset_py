import os

import numpy as np

import pandas as pd

from PIL import Image

import tensorflow as tf

import tensorflow.keras as keras

from functools import partial, wraps, reduce

from tensorflow.keras.layers import (Conv2D, MaxPool2D, BatchNormalization, InputLayer, LeakyReLU, Dense, 

Flatten, Dropout, ReLU, SeparableConv2D, GlobalAveragePooling2D)

from tensorflow.keras.regularizers import l2



tf.__version__
# path and label file creation



path_dict = {'val_normal':'/kaggle/input/chest-xray-pneumonia/chest_xray/val/NORMAL/',

            'val_pneumonia':'/kaggle/input/chest-xray-pneumonia/chest_xray/val/PNEUMONIA',

            'train_normal':'/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/',

            'train_pneumonia':'/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/',

            'test_normal':'/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/',

            'test_pneumonia':'/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/'

            }



data_paths = {}



for key, value in path_dict.items():

	paths_list = []

	for dirname, _, filenames in os.walk(value):

		for filename in filenames:

			paths_list.append(os.path.join(dirname, filename))

	data_paths[key] = paths_list



for key in data_paths:

    print(key," : ",len(data_paths[key]))





a = data_paths

n = 'val_normal'

p = 'val_pneumonia'



val_data = np.concatenate((

    np.concatenate((a[n], a[p]), axis=0).reshape(-1, 1),

    np.concatenate((

        np.zeros(len(a[n])),

        np.ones(len(a[p]))

    ), axis=0).reshape(-1, 1)

), axis=1)



n = 'train_normal'

p = 'train_pneumonia'



train_data = np.concatenate((

    np.concatenate((a[n], a[p]), axis=0).reshape(-1, 1),

    np.concatenate((

        np.zeros(len(a[n])),

        np.ones(len(a[p]))

    ), axis=0).reshape(-1, 1)

), axis=1)



n = 'test_normal'

p = 'test_pneumonia'



test_data = np.concatenate((

    np.concatenate((a[n], a[p]), axis=0).reshape(-1, 1),

    np.concatenate((

        np.zeros(len(a[n])),

        np.ones(len(a[p]))

    ), axis=0).reshape(-1, 1)

), axis=1)





# remove unwanted files



remove_train = []

remove_val = []

remove_test = []



for index, i in enumerate(train_data[:,0]):

    if (i.split('.')[1]) != 'jpeg':

        print(i, index)

        remove_train.append(index)



for index, i in enumerate(val_data[:,0]):

    if (i.split('.')[1]) != 'jpeg':

        print(i, index)

        remove_val.append(index)



for index, i in enumerate(test_data[:,0]):

    if (i.split('.')[1]) != 'jpeg':

        print(i, index)

        remove_test.append(index)



train_data = np.delete(train_data, remove_train, axis=0)

val_data = np.delete(val_data, remove_val, axis=0)

test_data = np.delete(test_data, remove_test, axis=0)
# adding more data to validation dataset from train

np.random.seed(10)



data = np.append(train_data, val_data, axis=0)

np.random.shuffle(data)



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(data[:,0], data[:,1], test_size=0.10, random_state=42)



print('Data shape after split: ', X_train.shape, X_val.shape)



unique_elements, counts_elements = np.unique(y_val, return_counts=True)

print('0 and 1 in val data :', counts_elements)



unique_elements, counts_elements = np.unique(y_train, return_counts=True)

print('0 and 1 in train data :', counts_elements)
@tf.function()

def read_decode_resize_img(path, label):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)

    original = img

    img = tf.image.random_flip_left_right(img)

    if tf.random.uniform(()) > 0.67:

        img = tf.image.random_contrast(img, 1, 2)

    if tf.random.uniform(()) > 0.67:

        tf.image.central_crop(img, 0.5)

    if tf.random.uniform(()) > 0.67:

        img = original

    return tf.image.resize(img, [224, 224]), label
@tf.function()

def read_decode_resize_img_no_aug(path, label):

    img = tf.io.read_file(path)

    img = tf.image.decode_jpeg(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)

    return tf.image.resize(img, [224, 224]), label
AUTOTUNE = tf.data.experimental.AUTOTUNE



train_ds_original = tf.data.Dataset.from_tensor_slices(

    (X_train, y_train.astype(float)))



train_ds = (train_ds_original

            .shuffle(len(X_train))

            .map(read_decode_resize_img, num_parallel_calls=AUTOTUNE)

            .batch(128)

            .prefetch(buffer_size=AUTOTUNE)

            .repeat(2))





val_ds = tf.data.Dataset.from_tensor_slices(

    (X_val, y_val.astype(float)))



val_ds = (val_ds

            .shuffle(len(X_val))

            .map(read_decode_resize_img_no_aug, num_parallel_calls=AUTOTUNE)

            .batch(128)

            .prefetch(buffer_size=AUTOTUNE))



test_ds = tf.data.Dataset.from_tensor_slices(

    (test_data[:, 0], test_data[:, 1].astype(float)))



test_ds = (test_ds

            .shuffle(len(test_data))

            .map(read_decode_resize_img_no_aug, num_parallel_calls=AUTOTUNE)

            .batch(32)

            .prefetch(buffer_size=AUTOTUNE))
import matplotlib.pyplot as plt

img_collection = next(iter(train_ds))

plt.figure(figsize=(10,10))

for i in range(10):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(tf.squeeze(img_collection[0][i,:,:,:]), cmap='binary')

    plt.xlabel(img_collection[1][i].numpy())

plt.show()
# Init



def compose(*funcs):

    if funcs:

        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)

    else:

        raise ValueError('Composition of empty sequence not supported.')



        

_MeNetConv2D = partial(SeparableConv2D,

                       kernel_size=3,

                       strides=1,

                       kernel_regularizer=l2(1e-3),

                       bias_regularizer=l2(1e-3),

                       padding='same')





@wraps(SeparableConv2D)

def MeNetConv2D(*args, **kwargs):

    return _MeNetConv2D(*args, **kwargs)





def MeNetConv2D_BN_Leaky(*args, **kwargs):

    return compose(

        MeNetConv2D(*args, **kwargs),

        BatchNormalization(),

         LeakyReLU(alpha=0.1))



_MeNetDense = partial(Dense,

                      activation='relu',

                      kernel_regularizer=l2(1e-3)

                      )





@wraps(Dense)

def MeNetDense(*args, **kwargs):

    return _MeNetDense(*args, **kwargs)





def MeNetDense_Dropout(*args, **kwargs):

    return compose(

        MeNetDense(*args, **kwargs),

        Dropout(0.4)

    )



# model



inputs = keras.Input(shape=(224, 224, 3))



conv1 = MeNetConv2D_BN_Leaky(filters=8, name='conv1')(inputs)

maxpool1 = MaxPool2D(name='maxpool1')(conv1)



conv2 = MeNetConv2D_BN_Leaky(filters=16, name='conv2')(maxpool1)

maxpool2 = MaxPool2D(name='maxpool2')(conv2)



conv3 = MeNetConv2D_BN_Leaky(filters=32, name='conv3')(maxpool2)

maxpool3 = MaxPool2D(name='maxpool3')(conv3)



conv4 = MeNetConv2D_BN_Leaky(filters=64, name='conv4')(maxpool3)

maxpool4 = MaxPool2D(name='maxpool4')(conv4)



conv5 = MeNetConv2D_BN_Leaky(filters=128, name='conv5')(maxpool4)

maxpool5 = MaxPool2D(name='maxpool5')(conv5)



conv6 = MeNetConv2D_BN_Leaky(filters=256, name='conv6')(maxpool5)

maxpool6 = MaxPool2D(name='maxpool6')(conv6)



flatten = Flatten()(maxpool6)

dropout = Dropout(0.3)(flatten)

dense1 = MeNetDense_Dropout(512)(dropout)

dense2 = MeNetDense_Dropout(128)(dense1)

outputs = Dense(1)(dense2)



model = keras.Model(inputs=inputs, outputs=outputs, name='MeNet_Model')
model.summary()
class WeightedBinaryCrossEntropy(keras.losses.Loss):

    

    def __init__(self, neg_weight=1.25, from_logits=True,

                 reduction=keras.losses.Reduction.AUTO,

                 name='weighted_binary_crossentropy'):

        super().__init__(reduction=reduction, name=name)

        self.neg_weight = neg_weight

        self.from_logits = from_logits

    

    @tf.function

    def call(self, y_true, y_pred):

        ce = tf.losses.binary_crossentropy(

            y_true, y_pred, from_logits=self.from_logits)[:,None]

        ce = ce*(y_true) + self.neg_weight*ce*(1-y_true)

        return ce
model.compile(optimizer='adam',

              loss=WeightedBinaryCrossEntropy(),

              metrics=['accuracy'])
def scheduler(epoch):

  if epoch <= 25:

    return 0.001

  if epoch > 25:

    return 0.0005



learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)



# pre_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss',

#                                            min_delta = 0.01, patience = 5, verbose = 1,mode = 'min',

#                                            baseline = 0.09, restore_best_weights=False)
with tf.device('/device:GPU:0'):

    history = model.fit(train_ds,validation_data=val_ds, epochs=35,

                       callbacks=[learning_rate_callback])
from matplotlib import pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper left')

plt.grid()

plt.show()
model.evaluate(test_ds)
X = []

y = []

for val in test_ds:

    X.append(val[0])

    y.append(val[1])



print(len(X), len(y))
recall = []

precision = []

confusion_matrix = []



for x, y_true in zip(X, y):

    y_pred = tf.math.sigmoid(model.predict(tf.squeeze(x)))

    y_pred = np.where(y_pred > 0.8, 1, 0)

    y_true = y_true.numpy().reshape(-1,1)

    

    m = tf.metrics.Recall()

    m.update_state(y_true, y_pred)

    recall.append(m.result().numpy())



    m = tf.metrics.Precision()

    m.update_state(y_true, y_pred)

    precision.append(m.result().numpy())

    

    m = tf.math.confusion_matrix(tf.squeeze(y_true), tf.squeeze(y_pred))

    confusion_matrix.append(m)
print('Recall :', np.mean(np.array(recall)),'Precision :', np.mean(np.array(precision)))



a = np.array(confusion_matrix)

result = np.zeros((2,2))

for i in range(20):

    result += confusion_matrix[i]



import seaborn as sn

classes = ['Normal','Pneumonia']

result = result.numpy()

f, ax = plt.subplots(figsize=(9, 6))

sn.heatmap(result, annot=True, ax=ax, cmap='viridis',

          xticklabels=classes, yticklabels=classes, fmt='d')

ax.set_ylim(len(result)-0.01, -0.01)
model.save('my_model') 
import shutil

shutil.make_archive('model', 'zip', '/kaggle/working/my_model')