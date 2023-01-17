# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import glob

import numpy as np

import pandas as pd 

import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

import cv2                 

from random import shuffle

from tqdm import tqdm  

import tensorflow as tf 

from tensorflow.keras import Model

from tensorflow.keras.utils import plot_model

import matplotlib.pyplot as plt

import matplotlib.cm as cm

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

from collections import Counter

import cv2

from tqdm import tqdm

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras import optimizers

from keras import backend as K

from keras.optimizers import SGD

from keras.layers.merge import concatenate

from sklearn.metrics import fbeta_score
glob.glob('/kaggle/input/diseasecotton/data/train/*')
TrainImage="/kaggle/input/diseasecotton/data/train"

TestImage="/kaggle/input/diseasecotton/data/val"

diseased_cotton_leaf_images = os.listdir(TrainImage + "/diseased cotton leaf")

diseased_cotton_plant_images = os.listdir(TrainImage + "/diseased cotton plant")

fresh_cotton_plant_images = os.listdir(TrainImage + "/fresh cotton plant")

fresh_cotton_leaf_images = os.listdir(TrainImage + "/fresh cotton leaf")
print(len(diseased_cotton_leaf_images), len(diseased_cotton_plant_images), len(fresh_cotton_plant_images), len(fresh_cotton_leaf_images))

NUM_TRAINING_IMAGES = len(diseased_cotton_leaf_images)+len(diseased_cotton_plant_images)+len(fresh_cotton_plant_images)+len(fresh_cotton_leaf_images)

print(NUM_TRAINING_IMAGES)
image_size = 224 

BATCH_SIZE = 16 

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



data_path = '/kaggle/input/diseasecotton'



train_datagen = ImageDataGenerator(rescale = 1./255,

                                   zoom_range = 0.2,

                                   rotation_range=15,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory(data_path + '/data/train',

                                                 target_size = (image_size, image_size),

                                                 batch_size = BATCH_SIZE,

                                                 class_mode = 'categorical',

                                                 shuffle=True)



testing_set = test_datagen.flow_from_directory(data_path + '/data/val',

                                            target_size = (image_size, image_size),

                                            batch_size = BATCH_SIZE,

                                            class_mode = 'categorical',

                                            shuffle = True)
print("train batch ", training_set.__getitem__(0)[0].shape)

print("test batch ", testing_set.__getitem__(0)[0].shape)
training_set.class_indices
labels = ["diseased cotton leaf","diseased cotton plant","fresh cotton leaf","fresh cotton plant"]
sample_data = testing_set.__getitem__(1)[0] 

sample_label = testing_set.__getitem__(1)[1] 
plt.figure(figsize=(15,12))

for i in range(15):

    plt.subplot(3, 6, i + 1)

    plt.axis('off')

    plt.imshow(sample_data[i])

    plt.title(labels[np.argmax(sample_label[i])])
def display_training_curves(training, validation, title, subplot):

    if subplot%10==1: # set up the subplots on the first call

        plt.subplots(figsize=(15,15), facecolor='#F0F0F0')

        plt.tight_layout()

    ax = plt.subplot(subplot)

    ax.set_facecolor('#F8F8F8')

    ax.plot(training)

    ax.plot(validation)

    ax.set_title('model '+ title)

    ax.set_ylabel(title)

    ax.set_xlabel('epoch')

    ax.legend(['train', 'valid.'])
# https://keras.io/examples/vision/grad_cam/

from tensorflow import keras



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):

    # First, we create a model that maps the input image to the activations

    # of the last conv layer

    last_conv_layer = model.get_layer(last_conv_layer_name)

    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)



    # Second, we create a model that maps the activations of the last conv

    # layer to the final class predictions

    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])

    x = classifier_input

    for layer_name in classifier_layer_names:

        x = model.get_layer(layer_name)(x)

    classifier_model = keras.Model(classifier_input, x)



    # Then, we compute the gradient of the top predicted class for our input image

    # with respect to the activations of the last conv layer

    with tf.GradientTape() as tape:

        # Compute activations of the last conv layer and make the tape watch it

        last_conv_layer_output = last_conv_layer_model(img_array)

        tape.watch(last_conv_layer_output)

        # Compute class predictions

        preds = classifier_model(last_conv_layer_output)

        top_pred_index = tf.argmax(preds[0])

        top_class_channel = preds[:, top_pred_index]



    # This is the gradient of the top predicted class with regard to

    # the output feature map of the last conv layer

    grads = tape.gradient(top_class_channel, last_conv_layer_output)



    # This is a vector where each entry is the mean intensity of the gradient

    # over a specific feature map channel

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))



    # We multiply each channel in the feature map array

    # by "how important this channel is" with regard to the top predicted class

    last_conv_layer_output = last_conv_layer_output.numpy()[0]

    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):

        last_conv_layer_output[:, :, i] *= pooled_grads[i]



    # The channel-wise mean of the resulting feature map

    # is our heatmap of class activation

    heatmap = np.mean(last_conv_layer_output, axis=-1)



    # For visualization purpose, we will also normalize the heatmap between 0 & 1

    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap, top_pred_index.numpy()
# https://keras.io/examples/vision/grad_cam/

def superimposed_img(image, heatmap):

    # We rescale heatmap to a range 0-255

    heatmap = np.uint8(255 * heatmap)



    # We use jet colormap to colorize heatmap

    jet = cm.get_cmap("jet")



    # We use RGB values of the colormap

    jet_colors = jet(np.arange(256))[:, :3]

    jet_heatmap = jet_colors[heatmap]



    # We create an image with RGB colorized heatmap

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((image_size, image_size))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)



    # Superimpose the heatmap on original image

    superimposed_img = jet_heatmap * 0.4 + image

    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img
#label smoothing https://www.linkedin.com/pulse/label-smoothing-solving-overfitting-overconfidence-code-sobh-phd/

def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):

    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)

    return loss
!pip install efficientnet

import efficientnet.tfkeras as efn
# training call backs 

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, epsilon=0.0001, patience=15, verbose=1)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
# https://stackoverflow.com/questions/42586475/is-it-possible-to-automatically-infer-the-class-weight-from-flow-from-directory

counter = Counter(training_set.classes)                          

max_val = float(max(counter.values()))       

class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}

class_weights
print(efn.EfficientNetB7(weights='imagenet').input_shape) 
# EXTRA In case you want to save the h5 file 

# please dont forget to change the model names.

checkpoint = ModelCheckpoint(

    'model.h5', 

    monitor='val_acc', 

    verbose=1, 

    save_best_only=True, 

    save_weights_only=False,

    mode='auto'

)
from keras import Input

from keras.layers import Conv2D, MaxPooling2D, Embedding, Reshape, Concatenate, SeparableConv2D

import tensorflow as tf

from keras.layers.normalization import BatchNormalization



inputs = Input(shape=(224, 224, 3))



# First conv block

x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(inputs)

x = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Second conv block

x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Third conv block

x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)



# Fourth conv block

x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(rate=0.2)(x)



# Fifth conv block

x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = SeparableConv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2, 2))(x)

x = Dropout(rate=0.2)(x)



# FC layer

x = Flatten()(x)

x = Dense(units=512, activation='relu')(x)

x = Dropout(rate=0.7)(x)

x = Dense(units=128, activation='relu')(x)

x = Dropout(rate=0.3)(x)

x = Dense(units=64, activation='relu')(x)

x = Dropout(rate=0.5)(x)



# Output layer

output = Dense(units=4, activation='softmax')(x)



# Creating model and compiling

model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer='RMSprop', loss=categorical_smooth_loss, metrics=['accuracy'])



# Callbacks

checkpoint = ModelCheckpoint(filepath='best_weights.hdf5', save_best_only=True, save_weights_only=True)

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=2, mode='max')

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=1, mode='min')
history = model.fit_generator(training_set,

                              validation_data=testing_set,

                              callbacks=[lr_reduce, es_callback],

                              epochs=20)
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 211)

display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
pretrained_efnet = efn.EfficientNetB7(input_shape=(image_size, image_size, 3), weights='noisy-student', include_top=False)



for layer in pretrained_efnet.layers:

  layer.trainable = False



x2 = pretrained_efnet.output

x2 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x2)

x2 = tf.keras.layers.Flatten(name="flatten_head")(x2)

x2 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x2)

x2 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x2)

model_out = tf.keras.layers.Dense(4, activation='softmax', name="predictions_head")(x2)



model_efnet = Model(inputs=pretrained_efnet.input, outputs=model_out)

model_efnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=categorical_smooth_loss,metrics=['accuracy'])

model_efnet.summary()
history_efnet = model_efnet.fit_generator(training_set,

                                          validation_data=testing_set,

                                          callbacks=[lr_reduce, es_callback],

                                          epochs=15)
display_training_curves(history_efnet.history['loss'], history_efnet.history['val_loss'], 'loss', 211)

display_training_curves(history_efnet.history['accuracy'], history_efnet.history['val_accuracy'], 'accuracy', 212)
#try

pretrained_densenet = tf.keras.applications.DenseNet121(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)



for layer in pretrained_densenet.layers[:-3]:

  layer.trainable = False



x1 = pretrained_densenet.output

x1 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x1)

x1 = tf.keras.layers.Flatten(name="flatten_head")(x1)

x1 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x1)

x1 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x1)

model_out = tf.keras.layers.Dense(4, activation='softmax', name="predictions_head")(x1)



model_densenet = Model(inputs=pretrained_densenet.input, outputs=model_out)

model_densenet.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),loss=categorical_smooth_loss,metrics=['accuracy'])



model_densenet.summary()
history_densenet = model_densenet.fit_generator(training_set,

                                                validation_data=testing_set,

                                                callbacks=[lr_reduce, es_callback],

                                                epochs=10)
#Best model
pretrained_densenet = tf.keras.applications.DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)



for layer in pretrained_densenet.layers:

  layer.trainable = False



x1 = pretrained_densenet.output

x1 = tf.keras.layers.AveragePooling2D(name="averagepooling2d_head")(x1)

x1 = tf.keras.layers.Flatten(name="flatten_head")(x1)

x1 = tf.keras.layers.Dense(64, activation="relu", name="dense_head")(x1)

x1 = tf.keras.layers.Dropout(0.5, name="dropout_head")(x1)

model_out = tf.keras.layers.Dense(4, activation='softmax', name="predictions_head")(x1)



model_densenet = Model(inputs=pretrained_densenet.input, outputs=model_out)

model_densenet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss=categorical_smooth_loss,metrics=['accuracy'])



model_densenet.summary()
plot_model(model_densenet, show_shapes=True, to_file='model_densenet.png')
history_densenet = model_densenet.fit_generator(training_set,

                                                validation_data=testing_set,

                                                callbacks=[lr_reduce, es_callback],

                                                epochs=30)
display_training_curves(history_densenet.history['loss'], history_densenet.history['val_loss'], 'loss', 211)

display_training_curves(history_densenet.history['accuracy'], history_densenet.history['val_accuracy'], 'accuracy', 212)
last_conv_layer_name = "conv5_block32_concat"

classifier_layer_names = [

    "bn",

    "relu",

    "averagepooling2d_head",

    "flatten_head",

    "dense_head",

    "dropout_head",

    "predictions_head"

]
# test image

file_path =  '/kaggle/input/diseasecotton/data/test/fresh cotton plant/dsd (405).jpg'

test_image = cv2.imread(file_path)

test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)

plt.imshow(test_image)

test_image = np.expand_dims(test_image,axis=0)
heatmap, top_index = make_gradcam_heatmap(test_image, model_densenet, last_conv_layer_name, classifier_layer_names)

print("predicted as", labels[top_index])
plt.matshow(heatmap)

plt.show()
plt.figure(figsize=(15,10))

for i in range(12):

    plt.subplot(3, 4, i + 1)

    plt.axis('off')

    heatmap, top_index = make_gradcam_heatmap(np.expand_dims(sample_data[i], axis=0), model_densenet, last_conv_layer_name, classifier_layer_names)

    img = np.uint8(255 * sample_data[i])

    s_img = superimposed_img(img, heatmap)

    plt.imshow(s_img)

    plt.title(labels[np.argmax(sample_label[i])] + " pred as: " + labels[top_index], fontsize=8)