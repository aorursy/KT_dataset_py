import sys
import os
import time
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import optimizers
from keras import callbacks
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import to_categorical, np_utils
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from tqdm import tqdm
%matplotlib inline
train = pd.read_csv('../input/nnfl-cnn-lab2/upload/train_set.csv')
train_image = []
test_image = []
for i in tqdm(range(train.shape[0])):
    img = image.load_img('../input/nnfl-cnn-lab2/upload/train_images/train_images/'+train['image_name'][i],target_size=(150,150,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)
y = np.array(train.drop(['image_name'],axis=1))
encoded_Y = LabelEncoder().fit(y).transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)
model = Sequential([
    Conv2D(64, (3, 3), input_shape=(150, 150, 3), padding="same", activation="relu"),
    Conv2D(64, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    Conv2D(256, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    Conv2D(512, (3, 3), padding="same", activation="relu"),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(6, activation="softmax")
])
model.compile(loss='binary_crossentropy', optimizer = optimizers.RMSprop(lr=0.00005), metrics = ['accuracy'])
model.fit(X, dummy_y, epochs = 21, batch_size = 50)
# model.save('model.h5')
test = pd.read_csv('../input/nnfl-cnn-lab2/upload/sample_submission.csv')
test_image = []
for i in tqdm(range(test.shape[0])):
    img = image.load_img('../input/nnfl-cnn-lab2/upload/test_images/test_images/'+test['image_name'][i],target_size=(150,150,3))
    img = image.img_to_array(img)
    test_image.append(img/255)
#from keras.models import load_model
#model = load_model("/kaggle/working/model.h5")
pred = np.argmax(model.predict(np.array(test_image)), axis=1)
test = pd.read_csv('../input/nnfl-cnn-lab2/upload/sample_submission.csv')
submission_df = test.copy()
submission_df["label"] = pred
submission_df.to_csv('submission.csv', index=False)