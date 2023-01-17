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
import tensorflow.keras as K

import numpy as np





input_tensor = K.Input(shape=(32, 32, 3))

(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

x_train = K.applications.densenet.preprocess_input(x_train)

y_train = K.utils.to_categorical(y_train, 10)

x_test = K.applications.densenet.preprocess_input(x_test)

y_test = K.utils.to_categorical(y_test, 10)



x_train = np.concatenate((x_train, np.flip(x_train, 2)), 0)

y_train = np.concatenate((y_train, y_train), 0)



model = K.applications.VGG16(include_top=False,

                                            pooling='max',

                                            input_tensor=input_tensor,

                                            weights='imagenet')



output = model.get_layer('block3_pool').output

x = K.layers.GlobalAveragePooling2D()(output)

x= K.layers.BatchNormalization()(x)

x = K.layers.Dense(256, activation='relu')(x)

x = K.layers.Dense(256, activation='relu')(x)

x = K.layers.Dropout(0.6)(x)

output = K.layers.Dense(10, activation='softmax')(x)



model = K.models.Model(model.input, output)



lrr = K.callbacks.ReduceLROnPlateau(

                                   monitor='val_acc',

                                   factor=.01,

                                   patience=3,

                                   min_lr=1e-5)



es = K.callbacks.EarlyStopping(monitor='val_acc',

                               mode='max',

                               verbose=1,

                               patience=10)



mc = K.callbacks.ModelCheckpoint('cifar10.h5',

                                 monitor='val_acc',

                                 mode='max',

                                 verbose=1,

                                 save_best_only=True)



model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['acc'])



history = model.fit(x_train, y_train,

                    validation_data=(x_test, y_test),

                    batch_size=128,

                    callbacks=[es, mc, lrr],

                    epochs=30,

                    verbose=1)



model.save('cifar10.h5')

import matplotlib.pyplot as plt

f,ax=plt.subplots(2,1, figsize=(10,15)) #Creates 2 subplots under 1 column



ax[0].plot(model.history.history['loss'],color='b',label='Training Loss')

ax[0].plot(model.history.history['val_loss'],color='r',label='Validation Loss')



#Next lets plot the training accuracy and validation accuracy

ax[1].plot(model.history.history['acc'],color='b',label='Training  Accuracy')

ax[1].plot(model.history.history['val_acc'],color='r',label='Validation Accuracy')