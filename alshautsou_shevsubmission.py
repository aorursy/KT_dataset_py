import time

import pandas as pd

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split



from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPool2D, Input, BatchNormalization, ReLU

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from tensorflow.keras.preprocessing.image import ImageDataGenerator



train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')



X = train.drop('label',axis=1)

y = train.label

    

X = X.values.astype('float32') / 255   

X = X.reshape(-1, 28, 28,1)

    

y = to_categorical(y)



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=1) 



def group_layers(model, filters, kernel, count):

    for i in range(count):

        model.add(Conv2D(filters, kernel, padding='same'))

        model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))

        model.add(ReLU())

        

custom_model = Sequential()

custom_model.add(Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1)))

custom_model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"))

custom_model.add(ReLU())



group_layers(custom_model, 32, (3,3), 2)



custom_model.add(AveragePooling2D(strides=2))

custom_model.add(Dropout(0.25))



group_layers(custom_model, 64, (3,3), 2)



custom_model.add(AveragePooling2D(strides=2))

custom_model.add(Dropout(0.25))

        

custom_model.add(Flatten())

custom_model.add(Dense(64))

custom_model.add(ReLU())

    

custom_model.add(BatchNormalization())

custom_model.add(Dense(10, activation='softmax'))



custom_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 

                                patience=3, 

                                verbose=1, 

                                factor=0.2, 

                                min_lr=1e-6)

    

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=300, restore_best_weights=True)



custom_model.fit(X_train, y_train, batch_size=64,

                    epochs=60,

                    validation_data=(X_valid, y_valid),

                    callbacks=[learning_rate_reduction, es],

                    verbose=2)



test = test.drop('id',axis=1)

X_test = test.values.astype('float32') / 255

X_test = X_test.reshape(-1, 28, 28,1)



sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sub = custom_model.predict(X_test)

sub = np.argmax(sub, axis=1)



sample_sub['label']= sub

sample_sub.to_csv('submission.csv',index=False)