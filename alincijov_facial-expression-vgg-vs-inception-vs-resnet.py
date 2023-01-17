import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Activation, add

from tensorflow.keras.utils import plot_model
label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']
df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)

df = df.iloc[1:]

df.head()
def get_train_data(df):

    x = []



    train = df['pixels'].to_numpy()



    for i in range(len(train)):

        x.append(train[i].split(' '))



    x = np.array(x)

    x = x.astype('float32').reshape(len(train), 48, 48, 1)



    return x
train = get_train_data(df)

labels = df['emotion'].to_numpy().astype('int')
plt.imshow(train[0].reshape(48, 48))
labels[0]
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.33, random_state=42)
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
X_train.shape
import time



class TimeHistory(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.times = []



    def on_epoch_begin(self, epoch, logs={}):

        self.epoch_time_start = time.time()



    def on_epoch_end(self, epoch, logs={}):

        self.times.append(time.time() - self.epoch_time_start)
time_callback_vgg = TimeHistory()

time_callback_incep = TimeHistory()

time_callback_resid = TimeHistory()
def vgg_block(layer_in, n_filters, n_conv):

    # add convolutional layers

    for _ in range(n_conv):

        layer_in = Conv2D(n_filters, (3,3), padding='same', activation='relu')(layer_in)

    # add max pooling layer

    layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)

    return layer_in
# define model input

visible = Input(shape=(48, 48, 1))

# add vgg module

layer = vgg_block(visible, 64, 2)

# add vgg module

layer = vgg_block(layer, 128, 2)

# add vgg module

layer = vgg_block(layer, 256, 4)



layer = Flatten()(layer)

layer = Dense(7, activation='softmax')(layer)



model_vgg = Model(inputs=visible, outputs=layer)

model_vgg.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_vgg.summary()

plot_model(model_vgg, show_shapes=True, to_file='vgg_block.png')
history_vgg = model_vgg.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[time_callback_vgg])
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):

    # 1x1 conv

    conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(layer_in)

    # 3x3 conv

    conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(layer_in)

    conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)

    # 5x5 conv

    conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(layer_in)

    conv5 = Conv2D(f3_out, (5,5), padding='same', activation='relu')(conv5)

    # 3x3 max pooling

    pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(layer_in)

    pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)

    # concatenate filters, assumes filters/channels last

    layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)

    return layer_out
# define model input

visible = Input(shape=(48, 48, 1))

# add inception block 1

layer = inception_module(visible, 64, 96, 128, 16, 32, 32)

# add inception block 1

layer = inception_module(layer, 128, 128, 192, 32, 96, 64)



layer = Flatten()(layer)

layer = Dense(7, activation='softmax')(layer)



model_inception = Model(inputs=visible, outputs=layer)

model_inception.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_inception.summary()

plot_model(model_inception, show_shapes=True, to_file='inception_block.png')
history_inception = model_inception.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[time_callback_incep])
def residual_module(layer_in, n_filters):

    merge_input = layer_in

    # check if the number of filters needs to be increase, assumes channels last format

    if layer_in.shape[-1] != n_filters:

        merge_input = Conv2D(n_filters, (1,1), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    # conv1

    conv1 = Conv2D(n_filters, (3,3), padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)

    # conv2

    conv2 = Conv2D(n_filters, (3,3), padding='same', activation='linear', kernel_initializer='he_normal')(conv1)

    # add filters, assumes filters/channels last

    layer_out = add([conv2, merge_input])

    # activation function

    layer_out = Activation('relu')(layer_out)

    return layer_out
# define model input

visible = Input(shape=(48, 48, 1))

# add vgg module

layer = residual_module(visible, 64)

# create model



layer = Flatten()(layer)

layer = Dense(7, activation='softmax')(layer)



model_residual = Model(inputs=visible, outputs=layer)

model_residual.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_residual.summary()

plot_model(model_residual, show_shapes=True, to_file='residual_block.png')
history_residual = model_residual.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[time_callback_resid])
plt.title("Accuracy")

plt.plot(history_residual.history['accuracy'], 'b')

plt.plot(history_inception.history['accuracy'], 'g')

plt.plot(history_vgg.history['accuracy'], 'r')
plt.title("Loss")

plt.plot(history_residual.history['loss'], 'b')

plt.plot(history_inception.history['loss'], 'g')

plt.plot(history_vgg.history['loss'], 'r')
plt.title("Value Loss")

plt.plot(history_residual.history['val_loss'], 'b')

plt.plot(history_inception.history['val_loss'], 'g')

plt.plot(history_vgg.history['val_loss'], 'r')
plt.title("Value Accuracy")

plt.plot(history_residual.history['val_accuracy'], 'b')

plt.plot(history_inception.history['val_accuracy'], 'g')

plt.plot(history_vgg.history['val_accuracy'], 'r')
plt.title("Time to train per epoch (seconds)")

plt.plot(time_callback_resid.times, 'b')

plt.plot(time_callback_incep.times, 'g')

plt.plot(time_callback_vgg.times, 'r')