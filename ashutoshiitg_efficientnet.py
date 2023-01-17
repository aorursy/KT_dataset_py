# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers.experimental import preprocessing

from tensorflow.keras.preprocessing import image



import pathlib

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
!ls ../input/plant-seedlings-classification
data_dir = '../input/plant-seedlings-classification/'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
train_dir = pathlib.Path(train_dir)

image_count = len(list(train_dir.glob('*/*.png')))

print(image_count)
batch_size = 32

img_height = 224

img_width = 224
train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, 

                                                              validation_split = 0.1,

                                                              subset = "training",

                                                              seed = 1337,

                                                              image_size = (img_height, img_width),

                                                              batch_size = batch_size)



val_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,

                                                            validation_split = 0.1,

                                                            subset = "validation",

                                                            seed = 1337,

                                                            image_size = (img_height, img_width),

                                                            batch_size = batch_size)
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(int(labels[i]))

        plt.axis("off")
class_names = train_ds.class_names

num_classes = len(class_names)

print(class_names)

print(num_classes)

#print(type(class_names))
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")
img_augmentation = Sequential(

    [

        preprocessing.RandomRotation(factor=0.15),

        preprocessing.RandomFlip(),

        preprocessing.RandomContrast(factor=0.1),

    ],

    name="img_augmentation",

)
plt.figure(figsize=(10, 10))

for image, label in train_ds.take(1):

    for i in range(9):

        ax = plt.subplot(3, 3, i + 1)

        aug_img = img_augmentation(image)

        plt.imshow(aug_img[i].numpy().astype("uint8"))

        plt.title("{}".format(class_names[label[i]]))

        plt.axis("off")
def input_preprocess(image, label):

    label = tf.one_hot(label, 12)

    return image, label
train_ds = train_ds.map(

    input_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE

)



val_ds = val_ds.map(input_preprocess)
strategy = tf.distribute.MirroredStrategy()
from tensorflow.keras.applications import EfficientNetB0

  

with strategy.scope():

    inputs = layers.Input(shape=(img_height, img_width, 3))

    x = img_augmentation(inputs)

    outputs = EfficientNetB0(include_top=True, weights=None, classes=12)(x)



    model = tf.keras.Model(inputs, outputs)

    model.compile(

        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]

    )



model.summary()



epochs = 40 # @param {type: "slider", min:10, max:100}

hist = model.fit(train_ds, epochs=epochs, batch_size = batch_size, validation_data=val_ds, verbose=2)
def plot_hist(hist):

    plt.plot(hist.history["accuracy"])

    plt.plot(hist.history["val_accuracy"])

    plt.title("model accuracy")

    plt.ylabel("accuracy")

    plt.xlabel("epoch")

    plt.legend(["train", "validation"], loc="upper left")

    plt.show()





plot_hist(hist)
test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file), file])

test = pd.DataFrame(test, columns=['filepath', 'file'])

print(test.head(2))

test.shape
from tqdm import tqdm



x_test = np.zeros((len(test), img_height, img_width, 3), dtype='float32')

for i, filepath in tqdm(enumerate(test['filepath'])):

    img = keras.preprocessing.image.load_img(os.path.join(data_dir, filepath), target_size = (img_height, img_width))

    x = keras.preprocessing.image.img_to_array(img)

    x_test[i] = x

print('test Images shape: {} size: {:,}'.format(x_test.shape, x_test.size))
one_hot_prediction = model.predict(x_test)

one_hot_prediction[0:2]
preds = np.argmax(one_hot_prediction, axis=1)

print(preds.shape)

print(preds)

prediction = [class_names[i] for i in preds]

print(len(prediction))

print(prediction)
sample_submission.head(5)
my_submission = pd.DataFrame({'file': test['file'], 'species': prediction})

print(my_submission.head(5))

print(my_submission.shape)
my_submission.to_csv('output.csv', index=False)