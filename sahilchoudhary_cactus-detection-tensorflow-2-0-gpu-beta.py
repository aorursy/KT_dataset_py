# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

from os import listdir

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

!pip install tensorflow==2.0.0-beta1



from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import tensorflow as tf

training_dir = '../input/train/train/'

training_imgs = listdir(training_dir)

num_training_imgs = len(training_imgs)

num_training_imgs
train_labels_df = pd.read_csv('../input/train.csv', index_col = 'id')

print("total entries : " + str(train_labels_df.size))

train_labels_df.head()
pd.value_counts(train_labels_df['has_cactus'])
def get_test_image_path(id):

    return training_dir + id



def draw_cactus_image(id, ax):

    path = get_test_image_path(id)

    img = mpimg.imread(path)

    plt.imshow(img)

    ax.set_title('Label: ' + str(train_labels_df.loc[id]['has_cactus']))



fig = plt.figure(figsize=(20,20))

for i in range(12):

    ax = fig.add_subplot(3, 4, i + 1)

    draw_cactus_image(training_imgs[i], ax)
train_image_path = [training_dir + ti for ti in training_imgs ]

train_image_labels = [ train_labels_df.loc[ti]['has_cactus'] for ti in training_imgs]





for i in range(10):

    print(train_image_path[i], train_image_labels[i])
def img_to_tensor(img_path):

    img_tensor = tf.cast(tf.image.decode_image(tf.io.read_file(img_path)), tf.float32)

    img_tensor /= 255.0 # normalized to [0,1]

    return img_tensor



img_to_tensor(train_image_path[0])
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_image_path, train_image_labels, test_size=0.2)



def process_image_in_record(path, label):

    return img_to_tensor(path), label



def build_training_dataset(paths, labels, batch_size = 32):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(process_image_in_record)

    ds = ds.shuffle(buffer_size = len(paths))

    ds = ds.repeat()

    ds = ds.batch(batch_size)

    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return ds



def build_validation_dataset(paths, labels, batch_size = 32):

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.map(process_image_in_record)

    ds = ds.batch(batch_size)

    return ds



train_ds = build_training_dataset(X_train, y_train)

validation_ds = build_validation_dataset(X_valid, y_valid)
mini_train_ds = build_training_dataset(X_train[:5], y_train[:5], batch_size=2)

# Fetch and print the first batch of 2 images

for images, labels in mini_train_ds.take(1):

    print(images)

    print(labels)
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(64, activation='relu'))

model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # because we are in a binary classification setup



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)
def plot_accuracies_and_losses(history):

    plt.title('Accuracy')

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.legend(['training', 'validation'], loc='upper left')

    plt.show()

    

    plt.title('Cross-entropy loss')

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.legend(['training', 'validation'], loc='upper left')

    plt.show()



plot_accuracies_and_losses(history)
cnn_model = tf.keras.Sequential()



cnn_model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))

cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))

cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))

cnn_model.add(tf.keras.layers.MaxPooling2D((2,2)))

cnn_model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))



cnn_model.add(tf.keras.layers.Flatten())

cnn_model.add(tf.keras.layers.Dense(64, activation='relu'))

cnn_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.summary()
history = cnn_model.fit(train_ds, epochs=20, steps_per_epoch=400, validation_data=validation_ds)
test_dir = '../input/test/test/'

test_imgs = listdir(test_dir)

print(len(test_imgs))

test_imgs[:5]
def path_to_numpy_array(path):

    tensor = img_to_tensor(path)

    array = tensor.numpy()

    return array



test_image_paths = [test_dir + ti for ti in test_imgs]

test_instances = np.asarray([path_to_numpy_array(tip) for tip in test_image_paths])



test_instances[:2]
predictions = cnn_model.predict(test_instances)

print(len(predictions))
submission_data = pd.DataFrame({'id': test_imgs, 'has_cactus': predictions.flatten()})

submission_data.head(20)
submission_data.to_csv('submission.csv', index=False)
!head submission.csv