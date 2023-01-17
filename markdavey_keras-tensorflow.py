import math
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import os

tf.logging.set_verbosity(tf.logging.INFO)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
output_dir = "."
OUTDIR = "."
OUTPUT_RESULT="submission.csv"
#Hyper prameters
INPUT_SHAPE = (28, 28, 1)
SCALE = 10
BATCH_SIZE=2048
ROWS_TO_READ=42000
ROWS_TO_SKIP=1
LEARNING_RATE=0.001
STEPS_TO_PROCESS=42000
num_classes=10

# input image dimensions
img_rows, img_cols = 28, 28
df = type('', (), {})()
print(datetime.now())
df.train = pd.read_csv('../input/train.csv', sep=",", skiprows=range(1,ROWS_TO_SKIP),nrows=ROWS_TO_READ)
print(datetime.now())
df.test = pd.read_csv('../input/test.csv', sep=",")
print(datetime.now())
#df.train.head(10)
df.train.describe()
df.labels = df.train['label'] #get target training values
df.labels = keras.utils.to_categorical(df.labels.values.astype('int32') , num_classes) # identify how many numbers to train
df.train = df.train.drop(columns='label', axis=1) #remove so only image data remains
df.train = df.train/255 # normalize data so rance is 0.0 to 1.0
df.test = df.test/255 # normalize data so rance is 0.0 to 1.0
df.train = df.train.values.astype('float32') # change all pixel values to float and numpy matrix
df.test = df.test.values.astype('float32')
df.train = df.train.reshape(df.train.shape[0], img_rows, img_cols, 1) #change change image correct shape 1x784 to 28by28
df.test = df.test.reshape(df.test.shape[0], img_rows, img_cols, 1)
df.trainX = df.train
df.labelsX = df.labels
datagen = ImageDataGenerator(
        rotation_range=17,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')
datagen.fit(df.trainX, augment=True)
i=15
for xbatch,ybatch in datagen.flow(df.trainX, df.labelsX, batch_size=BATCH_SIZE*200):
    df.train=np.append(df.train,xbatch)
    df.labels=np.append(df.labels,ybatch)
    i=i-1
    if i==0:
        break;
df.train = df.train.reshape(int(df.train.shape[0]/img_rows/img_cols), img_rows, img_cols, 1)
df.labels = df.labels.reshape(int(df.labels.shape[0]/10),10)

import matplotlib.pyplot as plt
plt.imshow(df.train[100000].reshape(28,28),cmap='gray')
plt.show()
checkpointer = ModelCheckpoint(filepath="./weights.hdf5",
                               verbose=1, save_best_only=True)
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=4, verbose=1, mode='auto')
df.X_train, df.X_test, df.y_train, df.y_test = train_test_split(df.train, df.labels,
                                                    test_size=0.1,
                                                    random_state=42)
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 input_shape=INPUT_SHAPE))
model.add(keras.layers.LeakyReLU(alpha=0.1))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.LeakyReLU(alpha=0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3, 3)))
model.add(keras.layers.LeakyReLU(alpha=0.1))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128))
model.add(keras.layers.LeakyReLU(alpha=0.1))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(df.X_train, df.y_train,
          batch_size=BATCH_SIZE,
          epochs=45,shuffle=True,
          validation_data=(df.X_test, df.y_test),
          callbacks=[checkpointer, earlystopper])
df.pred = model.predict(df.test)
x = [np.where(r==max(r))[0]  for r in df.pred]
flat_list = [item for sublist in x for item in sublist]
pred=pd.DataFrame({'Label':flat_list})
idx=pd.DataFrame({'ImageId':range(1,len(pred)+1)})
submission=pd.concat([idx,pred],axis=1)
submission.to_csv(OUTPUT_RESULT,index=False)
submission.describe()