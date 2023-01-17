import tensorflow as tf

tf.__version__
!pip install keras-tuner

import kerastuner
# Imports for Deep Learning

from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, MaxPooling2D

from keras.models import Sequential, Model

from keras.callbacks import Callback

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import VGG16

from keras import optimizers



# Imports to view data

import cv2

import numpy as np

from glob import glob

from matplotlib import pyplot as plt

from IPython.display import clear_output

from numpy import floor

import random



def plot_samples(letter):

    print("Samples images for letter " + letter)

    base_path = '../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/'

    img = base_path + letter + "_test.jpg"

    

    plt.figure(figsize=(16,16))

    plt.imshow(plt.imread(img))

    print(img)

    return cv2.imread(img)



plot_samples('A')
class PlotLearning(Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.acc = []

        self.val_acc = []

        self.fig = plt.figure()

        

        self.logs = []

        



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.acc.append(logs.get('acc'))

        self.val_acc.append(logs.get('val_acc'))

        self.i += 1

        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

        

        clear_output(wait=True)

        

        ax1.set_yscale('Log')

        ax1.plot(self.x, self.losses, label="loss")

        ax1.plot(self.x, self.val_losses, label="val_loss")

        ax1.legend()

        

        ax2.plot(self.x, self.acc, label="acc")

        ax2.plot(self.x, self.val_acc, label="val_acc")

        ax2.legend()

        

        plt.show()

        

        

plot = PlotLearning()
data_dir = "../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"

target_size = (128, 128)

target_dims = (128, 128, 3) # add channel for RGB

n_classes = 29

val_frac = 0.2

batch_size = 100



data_augmentor = ImageDataGenerator(

    samplewise_center=False, 

    samplewise_std_normalization=False, 

    validation_split=val_frac,

    featurewise_center=False, 

    featurewise_std_normalization=False,

    zca_whitening=False,

    zca_epsilon=1e-06,

    rotation_range=0.2,

    width_shift_range=0.0,

    height_shift_range=0.0,

    brightness_range=None,

    shear_range=0.0,

    zoom_range=0.0,

    channel_shift_range=0.2,

    fill_mode='nearest',

    cval=0.0,

    horizontal_flip=False,

    vertical_flip=False, 

    rescale=1./255,

    preprocessing_function=None, 

    data_format='channels_last', 

)



train_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=True, subset="training")

val_generator = data_augmentor.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, subset="validation")
model = Sequential([

    BatchNormalization(input_shape=target_dims),

    Conv2D(32, (3,3), kernel_regularizer="l2", activation='relu', padding='same'),

    MaxPooling2D((3, 3)),

    

    BatchNormalization(),

    Conv2D(64, (3,3), kernel_regularizer="l2", activation='relu', padding='same'),

    MaxPooling2D((3, 3)),

        

    BatchNormalization(),

    Conv2D(128, (3,3), kernel_regularizer="l2", activation='relu', padding='same'),

    MaxPooling2D((3, 3)),

    

    Flatten(),

    

    BatchNormalization(),

    Dense(256, kernel_regularizer="l2", activation='relu'),

    BatchNormalization(),

    Dense(128, kernel_regularizer="l2", activation='relu'),

    BatchNormalization(),

    Dense(64, kernel_regularizer="l2", activation='relu'),

    BatchNormalization(),

    Dense(n_classes, activation='softmax')

])



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])

model.summary()
model = VGG16(weights="imagenet", include_top=False, input_shape=target_dims)

for layer in model.layers[:-5]:

    layer.trainable = False



top_layers = model.output

top_layers = Flatten(input_shape=model.output_shape[1:])(top_layers)

top_layers = Dense(4096, activation='relu', kernel_initializer='random_normal')(top_layers)

top_layers = Dense(n_classes, activation='softmax')(top_layers)



model_final = Model(input=model.input, output=top_layers)



sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

model_final.compile(

    loss='categorical_crossentropy',

    optimizer=sgd, metrics=['acc'])

model_final.summary()
#model.fit_generator(train_generator, epochs=40, validation_data=val_generator, validation_steps=200, callbacks=[plot], workers=8, steps_per_epoch=2000)

model_final.fit_generator(train_generator, epochs=10, validation_data=val_generator, validation_steps=200, callbacks=[plot], workers=8, steps_per_epoch=2000)
img = cv2.resize(plot_samples('B'), (128, 128))[::-1]/255

print(np.argmax(model_final.predict(np.array([img]))))



img = cv2.resize(plot_samples('space'), (128, 128))[::-1]/255

print(np.argmax(model_final.predict(np.array([img]))))
model_final.save("ASL-model.h5")

print("Loss: {}\nValidation Accuracy: {}".format(*model.evaluate_generator(generator=val_generator, workers=8, steps=500)))
train_generator.class_indices.keys()