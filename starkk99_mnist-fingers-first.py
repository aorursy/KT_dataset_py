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
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head()
X_train = train.drop(['label'] ,axis = 1)
Y_train = train['label']
X_train = X_train/255.0
test = test/255.0
Y_train
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical
Y_train = to_categorical(Y_train , num_classes = 10)
Y_train
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32 , (3,3) ,input_shape = (28,28,1), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64 , (3,3) , activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512 , activation = 'relu' ),
#     tf.keras.layers.Dense(512 , activation = 'relu'),
#     tf.keras.layers.Dense(256 , activation = 'relu'),
    tf.keras.layers.Dense(10 , activation = 'softmax')
])
model.summary()
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(lr=0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])
from sklearn.model_selection import train_test_split
x_train , x_val , y_train , y_val = train_test_split(X_train , Y_train , test_size = 0.15 ,random_state = 5)#stratify = Y_train
x_val.shape
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,
        fill_mode = 'nearest')  # randomly flip images


datagen.fit(x_train)
history = model.fit_generator(datagen.flow(x_train , y_train , batch_size = 128), epochs = 20, validation_data = (x_val,y_val) , steps_per_epoch = x_train.shape[0]//128 , callbacks = [learning_rate_reduction])
model.evaluate(x_val , y_val)
Y_pred = model.predict(test)
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
print(Y_pred_classes)
test_df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')
test_df['Label'] = Y_pred_classes
test_df.to_csv('submissionMNIST5.csv' , index = False)
model8 = tf.keras.Sequential([
    tf.keras.layers.Dense(1024 , activation = 'relu' , input_shape = [784]),
    tf.keras.layers.Dense(512 , activation = 'relu'),
    tf.keras.layers.Dense(256 , activation = 'relu'),
#     tf.keras.layers.Dense(256 , activation = 'relu'),
#     tf.keras.layers.Dense(128 , activation = 'relu'),
    tf.keras.layers.Dense(10 , activation = 'softmax')
])
model8.summary()
from tensorflow.keras.optimizers import RMSprop

model8.compile(optimizer=RMSprop(lr = 0.0005),
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])
history8 = model8.fit(X_train , Y_train , epochs = 20 , batch_size = 128)
Y_pred8 = model8.predict_classes(test)
test_df['Label'] = Y_pred8
test_df.to_csv('submissionMNIST2.csv' , index = False)
