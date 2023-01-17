import cv2

import pandas as pd

import numpy as np



from tqdm import tqdm

from IPython.display import clear_output

from matplotlib import pyplot as plt



from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback

from keras.applications import MobileNetV2

from keras.optimizers import Adam, SGD

from keras.models import Sequential, Model

from keras.layers import Dense, SeparableConv2D, DepthwiseConv2D, Conv2DTranspose, Conv2D, Reshape, UpSampling2D, Activation, BatchNormalization, GlobalMaxPooling2D, MaxPooling2D, Flatten, Input, SeparableConv2D, Dropout, concatenate, multiply
data = pd.read_csv('../input/training/training.csv')

HEIGHT, WIDTH = 96, 96

data.keys()
data.fillna(method = 'ffill',inplace = True) # Credits: @Karan Jakhar
X = np.zeros((len(data['Image'])*2,HEIGHT,WIDTH))

image_dataset = data['Image']

for i in tqdm(range(len(data['Image']))): # Some Image Augmentation

    X[i] = np.array(image_dataset[i].split(), dtype='uint8').reshape((96,96))

    X[len(data['Image'])+i] = cv2.flip(np.array(image_dataset[i].split(), dtype='uint8').reshape((96,96)), 1)



del image_dataset

X = np.expand_dims(X, -1) / 255
data = data.drop('Image', axis=1).values.T[:4].T

y = np.zeros((len(data)*2,4))

for i in tqdm(range(len(data))):

    y[i] = data[i] / HEIGHT

    y[len(data)+i] = abs(data[i]-WIDTH) / HEIGHT

    

del data
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
s=8

filtersize=(3,3)



input_layer = Input(shape=(96,96,1))



hidden = Conv2D(s, (7,7))(input_layer)

hidden = Conv2D(s, (7,7))(hidden)



hidden = MaxPooling2D((2,2))(hidden)



hidden = Conv2D(2*s, (5,5))(hidden)

hidden = Conv2D(2*s, (5,5))(hidden)



hidden = MaxPooling2D((2,2))(hidden)



hidden = Conv2D(4*s, (3,3))(hidden)

hidden = Conv2D(4*s, (3,3))(hidden)



hidden = MaxPooling2D((2,2))(hidden)



hidden = Flatten()(hidden)



hidden = Dense(512, activity_regularizer='l2', kernel_regularizer='l2')(hidden)

hidden = Dense(128, activity_regularizer='l2', kernel_regularizer='l2')(hidden)

hidden = Dense(8, activity_regularizer='l2', kernel_regularizer='l2')(hidden)

hidden = BatchNormalization()(hidden)

hidden = Activation('relu')(hidden)



output = Dense(4, activation='sigmoid')(hidden)



model = Model(input_layer, output)



model.compile(

    optimizer=Adam(lr=0.0001),

    loss='mse',

    metrics=['accuracy']

)



model.summary()
datagen = ImageDataGenerator(

    featurewise_center=True,

    samplewise_center=True,

    featurewise_std_normalization=True,

    samplewise_std_normalization=True,

    rescale=1./255,

    shear_range=0,

    zoom_range=0,

    horizontal_flip=False,

    validation_split=0.0,

    brightness_range=[0.7, 1.3],   

)



datagen.fit(X)
model.fit_generator(

    datagen.flow(X, y, batch_size=64),

    validation_data=(X,y),

    steps_per_epoch=1000,

    callbacks=[plot],

    epochs=500,

)
model.save('Face LandMark Regressor.h5')