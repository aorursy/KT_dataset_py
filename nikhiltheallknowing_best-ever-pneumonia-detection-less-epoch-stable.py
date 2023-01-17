import numpy as np

import pandas as pd

import pylab as plt

import cv2

import os



from tqdm import tqdm

from IPython.display import clear_output

from scipy.stats import zscore

from IPython.display import SVG



from keras.utils.vis_utils import plot_model, model_to_dot

from keras.models import Sequential, Model

from keras.callbacks import Callback, EarlyStopping

from keras.optimizers import Adam, rmsprop

from keras.applications import Xception

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Conv2D, SeparableConv2D, Dense, Flatten, concatenate, multiply, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input
train_dir = '../input/chest_xray/chest_xray/train/'

val_dir = '../input/chest_xray/chest_xray/test/'



HEIGHT, WIDTH = 256,256
data_size = len(os.listdir(train_dir+'PNEUMONIA'))+len(os.listdir(train_dir+'NORMAL'))

X = np.zeros((data_size,HEIGHT,WIDTH,3))

y = np.zeros((data_size, 2))

i = 0

for subdir in ['NORMAL/', 'PNEUMONIA/']:

    for file in tqdm(os.listdir(train_dir+subdir)):

        try:

            X[i] = cv2.resize(cv2.imread(train_dir+subdir+file), (HEIGHT,WIDTH))/255

        except:

            pass # Error for reading .DS_Store file

        if subdir == 'NORMAL/':

            y[i] = np.array([1,0])

        else:

            y[i] = np.array([0,1])

        i += 1
data_size = len(os.listdir(val_dir+'PNEUMONIA'))+len(os.listdir(val_dir+'NORMAL'))

X_ = np.zeros((data_size,HEIGHT,WIDTH,3))

y_ = np.zeros((data_size, 2))

i = 0

for subdir in ['NORMAL/', 'PNEUMONIA/']:

    for file in tqdm(os.listdir(val_dir+subdir)):

        try:

            X_[i] = cv2.resize(cv2.imread(val_dir+subdir+file), (HEIGHT,WIDTH))/255

        except:

            pass # Error for reading .DS_Store file

        if subdir == 'NORMAL/':

            y_[i] = np.array([1,0])

        else:

            y_[i] = np.array([0,1])

        i += 1
X = X[::2]

y = y[::2]
datagen = ImageDataGenerator(

    horizontal_flip=True,

    rotation_range=15,

    featurewise_center=True,

    width_shift_range=0.2,

    height_shift_range=0.2,

    validation_split=0.1

)

datagen.fit(X)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

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
input_layer = Input((HEIGHT, WIDTH, 3))



hidden_layer1 = Conv2D(8, (3,3), activation='relu')(input_layer)

hidden_layer2 = SeparableConv2D(8, (3,3), activation='relu')(input_layer)

hidden_layer1 = Conv2D(8, (3,3), activation='relu')(hidden_layer1)

hidden_layer2 = SeparableConv2D(8, (3,3), activation='relu')(hidden_layer2)



hidden_layer1 = BatchNormalization()(hidden_layer1)

hidden_layer2 = BatchNormalization()(hidden_layer2)



hidden_layer1 = MaxPooling2D((3,3))(hidden_layer1)

hidden_layer2 = MaxPooling2D((3,3))(hidden_layer2)



hidden_layer1 = Conv2D(16, (3,3), activation='relu')(hidden_layer1)

hidden_layer2 = SeparableConv2D(16, (3,3), activation='relu')(hidden_layer2)

hidden_layer1 = Conv2D(16, (3,3), activation='relu')(hidden_layer1)

hidden_layer2 = SeparableConv2D(16, (3,3), activation='relu')(hidden_layer2)



hidden_layer1 = BatchNormalization()(hidden_layer1)

hidden_layer2 = BatchNormalization()(hidden_layer2)



hidden_layer1 = MaxPooling2D((2,2))(hidden_layer1)

hidden_layer2 = MaxPooling2D((2,2))(hidden_layer2)



hidden_layer1 = Conv2D(32, (5,5), activation='relu')(hidden_layer1)

hidden_layer2 = SeparableConv2D(32, (5,5), activation='relu')(hidden_layer2)

hidden_layer1 = Conv2D(32, (5,5), activation='relu')(hidden_layer1)

hidden_layer2 = SeparableConv2D(32, (5,5), activation='relu')(hidden_layer2)



hidden_layer1 = MaxPooling2D((4,4))(hidden_layer1)

hidden_layer2 = MaxPooling2D((4,4))(hidden_layer2)



hidden_layer = concatenate([hidden_layer1, hidden_layer2])



hidden_layer = GlobalAveragePooling2D()(hidden_layer)

hidden_layer = Dense(60, activation='relu')(hidden_layer)

hidden_layer = BatchNormalization()(hidden_layer)

hidden_layer = Dense(60, activation='relu')(hidden_layer)

hidden_layer = BatchNormalization()(hidden_layer)



output_layer = Dense(2, activation='softmax')(hidden_layer)



model = Model(input_layer, output_layer)

model.compile(

    optimizer=Adam(0.001),

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

model.summary()
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
model.fit_generator(

    datagen.flow(X, y, batch_size=64, subset='training'),

    epochs=20,

    validation_data=(X_,y_),

    steps_per_epoch=1000,

    callbacks=[plot, early_stopping],

    workers=8

)
test_img1 = cv2.resize(cv2.imread(train_dir+'NORMAL/IM-0427-0001.jpeg'), (HEIGHT,WIDTH))

test_img2 = cv2.resize(cv2.imread(train_dir+'PNEUMONIA/person755_bacteria_2659.jpeg'), (HEIGHT, WIDTH))

plt.imshow(test_img1)

plt.show()

plt.imshow(test_img2)

plt.show()

test_img1.shape