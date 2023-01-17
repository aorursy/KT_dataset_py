# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import os
from functools import partial

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import PIL.Image as image
import PIL

import tensorflow as tf
from sklearn.model_selection import train_test_split

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train_data_x = train_data.iloc[:,1:]
train_data_y = train_data.iloc[:,0]

# Take all the value for reshaping -1, dimension (28,28), colour 1
train_data_x = train_data_x.values.reshape(-1,28,28)
test_data_x = test_data.values.reshape(-1,28,28)

# Expected shape per image is 28,28 (width, height)
train_data_x[0].shape
plt.imshow(train_data_x[8], cmap='gray');
train_data_x = np.expand_dims(train_data_x, axis=-1)/255.0
test_data_x = np.expand_dims(test_data_x, axis=-1)/255.0
# Expected shape per image is 28,28,1 (width, height, colour)
test_data_x[0].shape
def data_augumentation(x_data, y_data, batch_size):
    """
    Data augmentation
    """
#     datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rotation_range=20, width_shift_range=0.1,height_shift_range=0.1, zoom_range=0.2)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5, width_shift_range=0.1,height_shift_range=0.1, zoom_range=0.1)
    datagen.fit(x_data)
    train_data = datagen.flow(x_data, y_data, batch_size=batch_size, shuffle=True)
    return train_data
class MyCallback(tf.keras.callbacks.Callback):
    """
    custom callback for epoch end
    """
    
    def on_epoch_end(self, epoch, logs={}):
        """
        some function
        """
        if logs['accuracy'] >= 0.999:
            self.model.stop_training = True
call = MyCallback()
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(32, (5,5), padding = 'same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.001),
    loss="sparse_categorical_crossentropy", metrics=['accuracy'])
TRAIN_SIZE=0.8
BATCH_SIZE=64
x_train, x_test, y_train, y_test = train_test_split(train_data_x, train_data_y, train_size=TRAIN_SIZE, random_state=10)
aug_train_data = data_augumentation(x_train, y_train, BATCH_SIZE)
hist = model.fit(aug_train_data, epochs=30, steps_per_epoch=int(len(x_train)/BATCH_SIZE), validation_data=(x_test, y_test), callbacks=[call])
plt.plot(hist.history['accuracy'][1:], label='train acc')
plt.plot(hist.history['val_accuracy'][1:], label='validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
aug_train_data = data_augumentation(train_data_x, train_data_y, BATCH_SIZE)
hist = model.fit(aug_train_data, epochs=30, steps_per_epoch=int(len(train_data_x)/BATCH_SIZE), callbacks=[call])
pred_y = model.predict(test_data_x)
pred_y = np.argmax(pred_y, axis=1)
# Generate Submission File 
out_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
out_df.Label = pred_y
out_df.to_csv("output.csv", index=False)