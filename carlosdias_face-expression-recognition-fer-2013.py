import numpy as np

import pandas as pd

from keras.utils import to_categorical, normalize



data = pd.read_csv('../input/fer2013.csv')



def get_dataset_from_usage(usage):

    df = data[data['Usage'] == usage]

    y = to_categorical(df['emotion'])

    X = np.array([values.split() for values in df['pixels']])

    X = X.astype(np.float).reshape(len(df), 48, 48, 1)

    return (X, y)



X_train, y_train = get_dataset_from_usage('Training')

X_test, y_test = get_dataset_from_usage('PublicTest')

X_validate, y_validate = get_dataset_from_usage('PrivateTest')



print('X_train:', X_train.shape)

print('y_train:', y_train.shape)



print('X_test:', X_test.shape)

print('y_test:', y_test.shape)



print('X_validate:', X_validate.shape)

print('y_validate:', y_validate.shape)

from keras.models import Sequential

from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint



emotion_header = [1, 2, 3, 4, 5, 6, 7]



model = Sequential()



model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',

                 input_shape=(48, 48, 1), data_format='channels_last',

                 kernel_regularizer=l2(0.01)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dropout(0.50))



model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dropout(0.50))



model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dropout(0.50))



model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))

model.add(Dropout(0.50))



model.add(Flatten())



model.add(Dense(512, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(len(emotion_header), activation='softmax'))



model.compile(loss='categorical_crossentropy',

              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),

              metrics=['accuracy'])



lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)



tensorboard = TensorBoard(log_dir='./logs')



early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')



checkpointer = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)



history = model.fit(X_train, y_train,

                    batch_size=64, epochs=100,

                    shuffle=True,

                    verbose=1, validation_data=(X_test, y_test),

                    callbacks=[lr_reducer, tensorboard, early_stopper, checkpointer])



score = model.evaluate(X_test, y_test, verbose=0)



print(score)



score = model.evaluate(X_validate, y_validate, verbose=0)



print(score)
!ls