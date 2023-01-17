import pandas as pd
from keras import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten,Activation,AveragePooling2D
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
import numpy as np
import keras
import keras.utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
# load the data in Data Frame
train_df = pd.read_csv('../input/train.csv')

train_df.head()
# Basic info on Data Frame
train_df.info()
# Initialise input data for CNN
input_data = np.zeros((train_df.shape[0],28,28))
# Getting Label from data Frame to np.array
Y_train = np.array(train_df['label'])
Y_train = keras.utils.to_categorical(Y_train, 10)
# method to convert an DataFrame row to matrix of n_rows
def conv_df_row_to_matrix(n_rows,arr):
    arr = arr / 255
    return arr.reshape(n_rows,-1)
# drop column label from data Frame 
train_np = train_df.drop('label',axis=1)
for i in range(train_df.shape[0]):
    input_data[i] = conv_df_row_to_matrix(28, np.array(train_np.iloc[i]))
X_train = input_data.reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],1)
print(X_train.shape)
print(Y_train.shape)
model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(32,(3,3), input_shape=(28,28,1), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2)))

model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(Conv2D(256,(3,3), padding='same'))
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))

model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
dataGen = ImageDataGenerator(rotation_range=15,width_shift_range=0.2,height_shift_range=0.2,
                             shear_range=0.15,zoom_range=[0.5,2],validation_split=0.2)
dataGen.fit(X_train)

train_generator = dataGen.flow(X_train, Y_train, batch_size=64, shuffle=True, 
                               seed=2, save_to_dir=None, subset='training')

validation_generator = dataGen.flow(X_train, Y_train, batch_size=64, shuffle=True, 
                               seed=2, save_to_dir=None, subset='validation')

filepath_val_acc="model_gen_cnn_acc.best.hdf5"
checkpoint_val_acc = ModelCheckpoint(filepath_val_acc, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint_val_acc]

history_inception_datagen = model.fit_generator(train_generator,
                                                steps_per_epoch = 600,
                                                epochs=30,
                                                validation_data = validation_generator,
                                                validation_steps = 150,
                                                callbacks = callbacks_list)
