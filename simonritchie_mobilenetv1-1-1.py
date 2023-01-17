import os

from keras.applications import MobileNetV2
from keras.applications import MobileNet
from keras.applications import VGG16
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn import preprocessing
from keras.utils import np_utils
os.listdir('../input/')
train_meta_df = pd.read_csv('../input/meta.csv')
train_meta_df[:3]
test_meta_df = pd.read_csv('../input/test_meta.csv')
test_meta_df[:3]
NUM_TRAIN = len(train_meta_df)
NUM_TEST = len(test_meta_df)
IMG_SIZE = 224
NUM_BATCH = 128
NUM_EPOCH = 40
NUM_TRAIN
NUM_TEST
X_train = np.memmap(
    filename='../input/X_train.npy', dtype=np.float16, 
    mode='r', shape=(NUM_TRAIN, IMG_SIZE, IMG_SIZE, 3) )
X_train[0]
y_train = train_meta_df['class'].values
le = preprocessing.LabelEncoder()
le.fit(y_train)
le.classes_
NUM_CLASSES = len(le.classes_)
NUM_CLASSES
y_train = le.transform(y_train)
y_train
y_train = np_utils.to_categorical(y=y_train, num_classes=NUM_CLASSES)
y_train.shape
X_test = np.memmap(
    filename='../input/X_test.npy', dtype=np.float16, 
    mode='r', shape=(NUM_TEST, IMG_SIZE, IMG_SIZE, 3))
X_test[0]
y_test = test_meta_df['class'].values
y_test = le.transform(y_test)
y_test = np_utils.to_categorical(y=y_test, num_classes=NUM_CLASSES)
y_test.shape
model = MobileNet(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    alpha=1.0, depth_multiplier=1, weights=None,
    classes=NUM_CLASSES)
model.compile(
    loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(
    x=X_train, y=y_train, batch_size=NUM_BATCH, epochs=NUM_EPOCH, 
    verbose=2, validation_data=(X_test, y_test))
model.save(filepath='./model.h5')
ls -lht

