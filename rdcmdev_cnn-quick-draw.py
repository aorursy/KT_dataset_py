import numpy as np
import pandas as pd

import _pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
%matplotlib inline

import scipy.ndimage
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras import regularizers

import warnings
warnings.filterwarnings('ignore')
!ls ../input
with open('../input/ammd2tp2018.pkl', 'rb') as input:
    dbdraw = pickle.load(input, encoding='latin1')
unpickler = pickle.Unpickler(input, encoding='latin1')
X_data, Y_data = [], []

for i in range(len(dbdraw)):
    X_data.append(dbdraw[i]['bitmap'].reshape(28, 28))
    X_data[i] = np.expand_dims(X_data[i], axis=2)
    Y_data.append(dbdraw[i]['label'])
X_data = np.array(X_data)

label_encoder = LabelEncoder()
Y_data_categoric = label_encoder.fit_transform(Y_data)
Y_data_categoric = pd.DataFrame(Y_data_categoric, columns=["label"])

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data_categoric, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.8, random_state=42)
train_labels = to_categorical(y_train)
test_labels = to_categorical(y_test)
val_labels = to_categorical(y_val)

len(train_labels), len(val_labels), len(test_labels)
# create data augmention
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

data_generator.fit(X_train)
patience = 100
batch_size = 1000
num_epochs = 1000

# callbacks
csv_logger = CSVLogger('cnn.log', append=False)

# EarlyStopping - https://keras.io/callbacks/#earlystopping
early_stop = EarlyStopping(monitor='val_loss', patience=patience, mode='min')

# ReduceLROnPlateau - ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.5, patience=(patience//4))

model_names = 'cnn_{val_acc:.2f}.hdf5'

# ModelCheckpoint - https://keras.io/callbacks/#modelcheckpoint
model_checkpoint = ModelCheckpoint(model_names,  monitor='val_acc', save_best_only=True, mode='max')

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]
model = Sequential()
model.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=3, padding='same', input_shape=(28, 28, 1), kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))

model.add(Convolution2D(filters=32, kernel_size=(5, 5), strides=3, padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(5, 5), padding='same'))

model.add(Flatten())
model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(.25))
model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu'))
model.add(Dropout(.25))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
model.summary()
## train network
history = model.fit_generator(data_generator.flow(X_train, train_labels, batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs, verbose=True, callbacks=callbacks,
                    validation_data=(X_val, val_labels))
test_loss, test_acc = model.evaluate(X_test, test_labels)
test_acc 
# https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
# https://stackoverflow.com/questions/51428696/keras-access-layer-parameter-of-pre-trained-model-to-freeze

for i in range(5):
    model.pop()
    
for layer in model.layers:
    layer.trainable = False
 
model.add(Dense(64, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(32, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
model.summary()    
## train network
history = model.fit_generator(data_generator.flow(X_train, train_labels, batch_size),
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=num_epochs, verbose=True, callbacks=callbacks,
                    validation_data=(X_val, val_labels))
test_loss, test_acc = model.evaluate(X_test, test_labels)
test_acc 