# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

from tensorflow import keras

from tensorflow.keras import layers

import tensorflow as tf

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')
data.columns
pix = data['pixels']
p = []

for i in pix:

    p.append(i.split(' '))
p = np.array(p)
p[0]
p.shape
p1 = np.reshape(p,(-1,48,48))
p1 = p1.astype('float32')
p1[15]
data[data['age']==26]
plt.figure(figsize=(10,10))



for index, image in enumerate(np.random.randint(0,10454,9)):

    plt.subplot(3, 3, index + 1)

    plt.imshow(p1[image])



plt.show()
y = data[['gender','ethnicity','age']]

X = p1
y_gender = np.array(y['gender'])

y_ethnicity = np.array(y['ethnicity'])

y_age = np.array(y['age'])
X.shape
def build_model(num_classes, activation='softmax', loss='sparse_categorical_crossentropy'):

    

    inputs = tf.keras.Input(shape=(img_height, img_width, 1))

    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)

    

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',loss=loss,metrics=['accuracy'])

    

    return model
from sklearn.model_selection import train_test_split



X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, train_size=0.7)

X_ethnicity_train, X_ethnicity_test, y_ethnicity_train, y_ethnicity_test = train_test_split(X, y_ethnicity, train_size=0.7)

X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X, y_age, train_size=0.7)