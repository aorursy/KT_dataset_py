import os

import warnings

import numpy as np

import pandas as pd

from PIL import Image

import tensorflow as tf

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.models import load_model

from keras import layers, optimizers 

from keras.preprocessing import image

from keras.models import Sequential, Model

from keras.optimizers import Adam

import keras

from sklearn.model_selection import train_test_split

from keras.layers import Input, GlobalAveragePooling2D, Dense, concatenate, AveragePooling2D

from keras.layers.convolutional import Conv2D

from keras.layers.core import Activation, Dropout

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True

session = tf.compat.v1.Session(config=config)
def plot_history(history):

    history_dict = history.history

    loss_values = history_dict['loss']

    val_loss_values = history_dict['val_loss']

    epochs = range(1, len(loss_values) + 1)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))



    ax1.plot(epochs, loss_values, 'bo',

             label='Training loss')

    ax1.plot(epochs, val_loss_values, 'r',

             label='Validation loss')

    ax1.set_xlabel('Epochs')

    ax1.set_ylabel('Loss')

    ax1.set_xscale('log')



    acc_values = history_dict['accuracy']

    val_acc_values = history_dict['val_accuracy']



    ax2.plot(epochs, acc_values, 'bo',

             label='Training acc')

    ax2.plot(epochs, val_acc_values, 'r',

             label='Validation acc')

    ax2.set_xlabel('Epochs')

    ax2.set_ylabel('Accuracy')

    ax2.set_xscale('log')



    plt.legend()

    plt.show()
test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

dig_mnist = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")

sample_submission = pd.read_csv("/kaggle/input/Kannada-MNIST/sample_submission.csv")
print("test.shape", test.shape)

print("train.shape", train.shape)

print("dig_mnist.shape", dig_mnist.shape)

print("sample_submission.shape", sample_submission.shape)
train.head(11)
X_train = train.loc[:, train.columns!='label'].values.astype('uint8')

print("X_train.shape", X_train.shape)

y_train = train['label'].values

X_train = X_train.reshape((X_train.shape[0],28,28))

print("X_train.shape", X_train.shape)

print("y_train.shape",X_train.shape)
test.head(11)
X_test = test.loc[:,  test.columns!='id'].values.astype('uint8')

print("X_test.shape", X_test.shape)

y_id = test['id'].values

X_test = X_test.reshape((X_test.shape[0],28,28))

print("X_test.shape", X_test.shape)

print("y_id.shape", y_id.shape)
n = np.random.randint(X_train.shape[0])

plt.imshow(Image.fromarray(X_train[n]))

plt.show()

print(f'This is a {y_train[n]}')
X_train = X_train[:,:,:,None]

X_test = X_test[:,:,:,None]
print("X_train.shape", X_train.shape)

print("X_test.shape", X_test.shape)
batch_size = 32

num_epochs = 50

num_samples = X_train.shape[0]

num_classes = np.unique(y_train).shape[0]

img_rows, img_cols = X_train[0,:,:,0].shape

classes = np.unique(y_train)
print("num_samples",num_samples)

print("num_classes",num_classes)

print("img_rows",img_rows)

print("img_cols",img_cols)

print("classes",classes)
y_train = np_utils.to_categorical(y_train, num_classes)

y_train.shape
X_train_norm = X_train.astype('float32')

X_test_norm = X_test.astype('float32')

X_train_norm /= 255

X_test_norm /= 255
learning_rate_reduction=ReduceLROnPlateau(monitor='val_loss',

                                          patience=5, 

                                          verbose=1,

                                          factor=0.2

                                         )
early_stopping = EarlyStopping(monitor='val_loss', 

                               mode='min', 

                               verbose=1, 

                               patience=10

                              )
model_check_point = ModelCheckpoint('model.h5', save_best_only=True)
class DenseNet:

    def __init__(self, input_shape=None, dense_blocks=3, dense_layers=-1, growth_rate=12, nb_classes=None,

                 dropout_rate=None, bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40):



        # Checks

        if nb_classes == None:

            raise Exception(

                'Please define number of classes (e.g. num_classes=10). This is required for final softmax.')



        if compression <= 0.0 or compression > 1.0:

            raise Exception('Compression have to be a value between 0.0 and 1.0.')



        if type(dense_layers) is list:

            if len(dense_layers) != dense_blocks:

                raise AssertionError('Number of dense blocks have to be same length to specified layers')

        elif dense_layers == -1:

            dense_layers = int((depth - 4) / 3)

            if bottleneck:

                dense_layers = int(dense_layers / 2)

            dense_layers = [dense_layers for _ in range(dense_blocks)]

        else:

            dense_layers = [dense_layers for _ in range(dense_blocks)]



        self.dense_blocks = dense_blocks

        self.dense_layers = dense_layers

        self.input_shape = input_shape

        self.growth_rate = growth_rate

        self.weight_decay = weight_decay

        self.dropout_rate = dropout_rate

        self.bottleneck = bottleneck

        self.compression = compression

        self.nb_classes = nb_classes

        

    def build_model(self):

        img_input = Input(shape=self.input_shape, name='img_input')

        nb_channels = self.growth_rate

        

        x = Conv2D(2*self.growth_rate, (3,3), 

                   padding='same', strides = (1,1), 

                   kernel_regularizer=keras.regularizers.l2(self.weight_decay))(img_input)

        

        for block in range(self.dense_blocks-1):

            x, nb_channels = self.dense_block(x, self.dense_layers[block], nb_channels, self.growth_rate,

                                              self.dropout_rate, self.bottleneck, self.weight_decay)

            

            x = self.transition_layer(x, nb_channels, self.dropout_rate, self.compression, self.weight_decay)

            nb_channels = int(nb_channels*self.compression)

            

        x, nb_channels = self.dense_block(x, self.dense_layers[-1], nb_channels, self.growth_rate, self.dropout_rate, self.weight_decay)

        

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)

        prediction = Dense(self.nb_classes, activation='softmax')(x)

        

        return Model(inputs=img_input, outputs=prediction, name='densenet')

        

    def dense_block(self, x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4):

        for i in range(nb_layers):

            cb = self.convolution_block(x, growth_rate, dropout_rate, bottleneck)

            nb_channels += growth_rate

            x = concatenate([cb,x])

            

        return x, nb_channels

    

    def convolution_block(self, x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4):       



        # Bottleneck

        if bottleneck:

            bottleneckWidth = 4

            x = BatchNormalization()(x)

            x = Activation('relu')(x)

            x = Conv2D(nb_channels * bottleneckWidth, (1, 1),

                                     kernel_regularizer=keras.regularizers.l2(weight_decay))(x)

            # Dropout

            if dropout_rate:

                x = Dropout(dropout_rate)(x)



        # Standard (BN-ReLU-Conv)

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Conv2D(nb_channels, (3, 3), padding='same')(x)



        # Dropout

        if dropout_rate:

            x = Dropout(dropout_rate)(x)



        return x



    def transition_layer(self, x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4):

        x = BatchNormalization()(x)

        x = Activation('relu')(x)

        x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',

                                 kernel_regularizer=keras.regularizers.l2(weight_decay))(x)



        # Adding dropout

        if dropout_rate:

            x = Dropout(dropout_rate)(x)



        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x
densenet = DenseNet((28,28,1), nb_classes=10, depth=21)
model = densenet.build_model()

model_optimizer = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])

model.summary()
history = model.fit(X_train_norm, 

                     y_train, 

                     batch_size = batch_size,  

                     epochs = num_epochs, 

                     validation_split = 0.1,

                     shuffle = True,

                     callbacks = [learning_rate_reduction, early_stopping]

                    )
plot_history(history)
model.save('model.h5')
# model = load_model('/kaggle/input/kannada-mnist-simpe-densenet-in-keras-weight/model.h5')
pred = model.predict(X_test_norm)
pred=np.argmax(pred, axis=1)
sample_submission['label'] = pred
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)