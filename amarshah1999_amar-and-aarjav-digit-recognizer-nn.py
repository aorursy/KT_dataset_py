# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')
X_train = pd.DataFrame(train.drop(['label'], axis = 1))
y_train = pd.DataFrame(train['label'])
X_train = (X_train.values/255).reshape(-1,28,28,1)

X_test = (test.values/255).reshape(-1,28,28,1)

"""
for i in range(0,10):
    g = plt.imshow(X_train[i][:,:,0], cmap = 'gray_r')
    plt.title(label = 'Digit: ' + str(y_train.values[i][0]))
    plt.show()
"""
y_train = pd.DataFrame(to_categorical(y_train, 10))

# detect and init the TPU
# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
# tf.config.experimental_connect_to_cluster(tpu)
# tf.tpu.experimental.initialize_tpu_system(tpu)

# # instantiate a distribution strategy
# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


data_generator_with_aug = tf.python.keras.preprocessing.image.ImageDataGenerator(
    width_shift_range = 0.1,height_shift_range = 0.1,rotation_range = 20)
data_generator_with_aug.fit(X_train)
# instantiating the model in the strategy scope creates the model on the TPU
"""with tpu_strategy.scope():
    model = Sequential()
    model.add(Conv2D(20, kernel_size=5, activation = 'relu', padding = 'Same'))
    model.add(Conv2D(20, kernel_size=5, activation = 'relu', padding = 'Same'))
    #model.add(Dropout(0.5))
    model.add(Conv2D(20, kernel_size=5, activation = 'relu', padding = 'Same'))
    model.add(Conv2D(20, kernel_size=5, activation = 'relu', padding = 'Same'))
    
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(10, activation = 'softmax'))
 # define your model normally
    model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])
"""


model = Sequential()
model.add(Conv2D(32, kernel_size=5, activation = 'relu', padding = 'Same', input_shape = (28,28,1)))
model.add(Conv2D(32, kernel_size=5, activation = 'relu', padding = 'Same'))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=5, activation = 'relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size=5, activation = 'relu', padding = 'Same'))
model.add(Dropout(0.5))

model.add(Conv2D(64, kernel_size=5, activation = 'relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size=5, activation = 'relu', padding = 'Same'))
model.add(Dropout(0.5))


model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = "categorical_crossentropy", optimizer = "RMSprop", metrics = ["accuracy"])
batch_size = 500
X_training,X_valid,y_training, y_valid = train_test_split(X_train, y_train, test_size = 0.1)



reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                              patience=3, min_lr=0.00001)

#model.fit_generator(data_generator_with_aug.flow(X_training,y_training, batch_size=batch_size), epochs = 50, steps_per_epoch = 75, validation_data = (X_valid, y_valid), callbacks = [reduce_lr])

"""
trained_weights = model.get_weights()
K.clear_session()
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2,
                              patience=5, min_lr=0.001)
model.set_weights(trained_weights)
model.fit(X_training, y_training,
                      batch_size = batch_size,
                      validation_data=(X_valid, y_valid),
                      epochs=50,
                      verbose=1, callbacks = [reduce_lr])
"""
# preds = model.predict(X_test, verbose = 1)
# preds = np.argmax(preds,axis = 1)
# test_submission = pd.DataFrame({'ImageId':range(1,28001),'Label': preds})
# test_submission.to_csv('submission_12.csv', index = False)