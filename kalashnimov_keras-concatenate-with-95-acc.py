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
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator 

from tensorflow.keras.layers import Input, Conv2D, Dense, Activation, Flatten, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Concatenate

from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.utils import to_categorical

from tensorflow.keras import regularizers, optimizers

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
img_rows = 28

img_cols = 28

num_classes = 10



def data_prep(raw):

    out_y = to_categorical(raw.label, num_classes)

    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
X_train, y_train = data_prep(train)

X_test, y_test = data_prep(test)
print("X_train.shape:", X_train.shape)

print("y_train.shape", y_train.shape)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose = 1, patience=20)

mc = ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', mode='max', verbose = 1, save_best_only=True)
# build model

i = Input(shape=(28,28,1))



x = Conv2D(32, kernel_size=(3, 3), strides=1, padding ="same")(i)

x = LeakyReLU(alpha=0.1)(x)

x = BatchNormalization()(x)

x = Conv2D(32, (3, 3), padding ="same")(x)

x = LeakyReLU(alpha=0.1)(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Dropout(0.5)(x)



x = Conv2D(64, kernel_size=(3, 3), strides=1, padding ="same")(x)

x = LeakyReLU(alpha=0.1)(x)

x = BatchNormalization()(x)

x = Conv2D(64, (3, 3), padding ="same")(x)

x = LeakyReLU(alpha=0.1)(x)

x = BatchNormalization()(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Dropout(0.5)(x)



x = Flatten()(x)



y = Conv2D(32, kernel_size=(3, 3), strides=1, padding ="same")(i)

y = LeakyReLU(alpha=0.1)(y)

y = BatchNormalization()(y)

y = Conv2D(32, (3, 3), padding ="same")(y)

y = LeakyReLU(alpha=0.1)(y)

y = BatchNormalization()(y)

y = AveragePooling2D(pool_size=(2,2))(y)

y = Dropout(0.5)(y)



y = Conv2D(64, kernel_size=(3, 3), strides=1, padding ="same")(y)

y = LeakyReLU(alpha=0.1)(y)

y = BatchNormalization()(y)

y = Conv2D(64, (3, 3), padding ="same")(y)

y = LeakyReLU(alpha=0.1)(y)

y = BatchNormalization()(y)

y = AveragePooling2D(pool_size=(2,2))(y)

y = Dropout(0.5)(y)



y = Flatten()(y)



xy = Concatenate()([x, y])

xy = Dropout(0.5)(xy)

xy = Dense(256, activation='relu')(xy)

xy = Dropout(0.5)(xy)

xy = Dense(10, activation='softmax')(xy)



model = Model(inputs = i, outputs = xy, name = 'max_avg')
model.summary()
# run model

model.compile(optimizer='Adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, 

          validation_data=(X_test, y_test), 

          batch_size = 128, epochs = 200, 

          callbacks = [es, mc])
def plot_model(history): 

    fig, axs = plt.subplots(1,2,figsize=(16,5)) 

    # summarize history for accuracy

    axs[0].plot(history.history['accuracy'], 'c') 

    axs[0].plot(history.history['val_accuracy'], 'm') 

    axs[0].set_title('Model Accuracy')

    axs[0].set_ylabel('Accuracy') 

    axs[0].set_xlabel('Epoch')

    axs[0].legend(['train', 'validate'], loc='upper left')

    # summarize history for loss

    axs[1].plot(history.history['loss'], 'c') 

    axs[1].plot(history.history['val_loss'], 'm') 

    axs[1].set_title('Model Loss')

    axs[1].set_ylabel('Loss') 

    axs[1].set_xlabel('Epoch')

    axs[1].legend(['train', 'validate'], loc='upper right')

    plt.show()
plot_model(model.history)
saved_model = load_model('best_cnn_model.h5')

test_loss, test_acc = saved_model.evaluate(X_test,  y_test, verbose=0)

print('Test Accuracy:', round(test_acc, 2))