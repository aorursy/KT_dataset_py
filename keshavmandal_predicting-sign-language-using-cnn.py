# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

test_df = pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
train_df.head()
plt.figure(figsize = (10,10))

sns.countplot(train_df['label'])
#Dropping lables from Training and Test Dataset and assigning to a new variable.

y_train = train_df['label']

y_test = test_df['label']



train_df.drop('label', axis = 1, inplace = True)

test_df.drop('label', axis = 1, inplace = True)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()

y_train = label_binarizer.fit_transform(y_train)

y_test = label_binarizer.fit_transform(y_test)
y_train.shape
x_train = train_df.values.reshape(-1,28,28,1)

x_test = test_df.values.reshape(-1,28,28,1)
x_train.shape
# Normalize the data

x_train = x_train / 255.

x_test = x_test / 255.
from keras_preprocessing.image import ImageDataGenerator

training_datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        shear_range=0.2,

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
training_datagen.fit(x_train)
import tensorflow as tf

model = tf.keras.models.Sequential([

    # Note the input shape is the desired size of the image 28X28 with 1 byte color

    # This is the first convolution

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1),padding = 'same'),

    tf.keras.layers.MaxPooling2D(2, 2),

    # The second convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The third convolution

    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    # The fourth convolution

    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding = 'same'),

    tf.keras.layers.MaxPooling2D(2,2),

    # Flatten the results to feed into a DNN

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.2),

    # 512 neuron hidden layer

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(24, activation='softmax')

])
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
history = model.fit(training_datagen.flow(x_train,y_train, batch_size = 128), epochs=25, steps_per_epoch=20, validation_data = (x_test,y_test), verbose = 1, callbacks = [learning_rate_reduction])#, validation_steps=3)
print("Model Accuracy - " , model.evaluate(x_test,y_test)[1]*100 , "%")
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training Accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend(loc=0)

plt.figure()



plt.plot(epochs, loss, 'r', label='Training Loss')

plt.plot(epochs, val_loss, 'b', label='Validation Loss')

plt.title('Training and Validation Loss')

plt.legend(loc=0)

plt.figure()





plt.show()