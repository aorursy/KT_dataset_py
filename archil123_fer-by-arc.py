import sys, os

import pandas as pd

import numpy as np

import cv2

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from keras.losses import categorical_crossentropy

from keras.optimizers import Adam

from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint

from keras.models import load_model
import pandas as pd

training = pd.read_csv("../input/training.csv")

validation = pd.read_csv("../input/validation.csv")
BASEPATH = './'

sys.path.insert(0, BASEPATH)

os.chdir(BASEPATH)

MODELPATH = './model2.h5'
num_features = 64

num_labels = 7

batch_size = 64

epochs = 100

width, height = 48, 48
train_pixels = training['pixels'].tolist()
train_faces = []

for pixel_sequence in train_pixels:

    train_face = [int(train_pixels) for train_pixels in (pixel_sequence.strip()).split(' ')] # 2

    train_face = np.asarray(train_face).reshape(width, height) # 3

    

    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.

    # face = face / 255.0 # 4

    # face = cv2.resize(face.astype('uint8'), (width, height)) # 5

    train_faces.append(train_face.astype('float32'))



train_faces = np.asarray(train_faces)

train_faces = np.expand_dims(train_faces, -1) # 6



train_emotions = pd.get_dummies(training['class']).as_matrix()
test_pixels = validation['pixels'].tolist() # 1



test_faces = []

for pixel_sequence in test_pixels:

    test_face = [int(test_pixels) for test_pixels in (pixel_sequence.strip()).split(' ')] # 2

    test_face = np.asarray(test_face).reshape(width, height) # 3

    

    # There is an issue for normalizing images. Just comment out 4 and 5 lines until when I found the solution.

    # face = face / 255.0 # 4

    # face = cv2.resize(face.astype('uint8'), (width, height)) # 5

    test_faces.append(test_face.astype('float32'))



test_faces = np.asarray(test_faces)

test_faces = np.expand_dims(test_faces, -1) # 6

test_emotions = pd.get_dummies(validation['class']).as_matrix() # 7

x_train , y_train , x_test , y_test = train_faces , train_emotions , test_faces , test_emotions
model = Sequential()



model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))

model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())



model.add(Dense(2*2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*2*num_features, activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2*num_features, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(num_labels, activation='softmax'))
model.compile(loss=categorical_crossentropy,

              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),

              metrics=['accuracy'])
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)

tensorboard = TensorBoard(log_dir='./logs')

early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

checkpointer = ModelCheckpoint(MODELPATH, monitor='val_loss', verbose=1, save_best_only=True)
model.fit(np.array(X_train), np.array(y_train),

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(np.array(X_test), np.array(y_test)),

          shuffle=True,

          callbacks=[lr_reducer,early_stopper, checkpointer])