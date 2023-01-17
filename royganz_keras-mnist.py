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
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dropout, MaxPool2D, Input, ReLU, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
# Load
train = pd.read_csv('../input/digit-recognizer/train.csv')
test = pd.read_csv('../input/digit-recognizer/test.csv')

X_train = train.drop(labels = ["label"], axis = 1)
Y_train = train['label']

X_test = test

# Scale
X_train, X_test = X_train.astype('float32'), X_test.astype('float32')
X_train = (X_train / 255.) * 2 - 1
X_test = (X_test / 255.) * 2 - 1
# one hot encode target values
Y_train = to_categorical(Y_train)


# Split to train and validation
X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state = 2)

# to_numpy + Reshape
X_train, X_test, X_val = X_train.to_numpy().reshape(-1,28,28,1), X_test.to_numpy().reshape(-1,28,28,1), X_val.to_numpy().reshape(-1,28,28,1)

_, im_h, im_w, chnls = X_train.shape
print(f'image height: {im_h}, image width: {im_w}, channels: {chnls}')
# The network will be written using keras functional API coding

# creates a cnn block as described above
def cnn_block(inputs, filters, kernel_size=3):
  # first conv + relu
  block_output = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(inputs)
  # second conv + relu
  block_output = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(block_output)
  # max pooling to reduce spatial dimensions
  block_output = MaxPool2D(pool_size=(2, 2))(block_output)
  return block_output

kernels = [32, 64, 128]

i = Input(shape=(im_h, im_w, chnls))
# Feature extraction
for block_num, kernel in enumerate(kernels):
  if block_num != 0:
    inputs = output
  else:
    inputs = i
  output = cnn_block(inputs, kernel)
# Classification
_, h,w,ch = output.shape
output = Flatten()(output)  # reshape the output from image shape into vector shape
output = Dense(h * w * ch, activation='relu')(output)
output = Dropout(0.5)(output)
output = Dense(10, activation='softmax')(output)

# Generate the model
CNN_classifier = Model(i, output)
# Compile the mode

CNN_classifier.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
plot_model(model=CNN_classifier, to_file='CNN_classifier_arch.png', show_shapes=True,show_layer_names=False)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# a callback is an object that performs tasks during the fit process
my_callbacks = [EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5')]


history = CNN_classifier.fit(x=X_train, y=y_train, batch_size=128, epochs=100, validation_data=(X_val, y_val),callbacks=my_callbacks)
output = CNN_classifier.predict(X_test)

# select the indix with the maximum probability
output = np.argmax(output, axis = 1)

output = pd.Series(output, name="Label")

submission = pd.concat([pd.Series(range(1,28001), name = "ImageId"), output], axis = 1)

submission.to_csv("submission.csv", index=False)