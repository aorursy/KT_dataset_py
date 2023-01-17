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
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau

def get_data_train(filename):
    with open(filename) as training_file:
        file = csv.reader(training_file, delimiter = ",")
        images = []
        labels = []
        ignore = 1
        for row in file:
            if ignore == 1:
                ignore = 0
                continue
            labels.append(row[0])
            images.append(np.array_split(row[1:],28))
    return np.array(images).astype("int32"), np.array(labels).astype("int32")
def get_data_test(filename):
    with open(filename) as training_file:
        file = csv.reader(training_file, delimiter = ",")
        images = []
        ignore = 1
        for row in file:
            if ignore == 1:
                ignore = 0
                continue
            images.append(np.array_split(row,28))
    return np.array(images).astype("int32")

train_path = '/kaggle/input/digit-recognizer/train.csv'
test_path = '/kaggle/input/digit-recognizer/test.csv'

train_images, train_labels = get_data_train(train_path)
test_images = get_data_test(test_path)

train_datagen = ImageDataGenerator(rescale = 1./255,
      rotation_range=15,
      width_shift_range=0.1,
      height_shift_range=0.1,
      shear_range=0.1,
      zoom_range=0.1,
      horizontal_flip=False,
      fill_mode='nearest'
    )
#final model
final_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (5,5), activation=tf.nn.relu,padding='Same',input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu,padding = 'Same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

final_model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])

final_model.summary()

train_labels_cat = to_categorical(train_labels)
train_images = np.expand_dims(train_images, axis=3)
test_images = test_images/255.0

learning_rate_reduction_final = ReduceLROnPlateau(monitor='acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.000003)


history = final_model.fit(train_datagen.flow(train_images, train_labels_cat, batch_size=64),
                    epochs = 30,
                    verbose = 1,
                   callbacks=[learning_rate_reduction_final])

%matplotlib inline
acc = history.history['acc']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()
test_images = np.expand_dims(test_images, axis=3)

results = final_model.predict(test_images)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

print(results.shape)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
get_ipython().run_cell_magic('javascript', '', '<!-- Save the notebook -->\nIPython.notebook.save_checkpoint();')
