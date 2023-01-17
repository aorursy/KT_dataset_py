# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python https://www.kaggle.com/uysimty/get-start-image-classification

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
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
train.head(10)
train.isnull().any().describe()
y=train['label']

y_asli=y

X=train.drop('label', axis=1)
X=X.values.reshape(-1,28,28,1)
from keras.utils.np_utils import to_categorical



y=to_categorical(y, num_classes=10)
from sklearn.model_selection import train_test_split

#X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=0)
import matplotlib.pyplot as plt

plt.imshow(X_train[3][:,:,0])
from tensorflow.keras.preprocessing.image import ImageDataGenerator

augmentation_generate = ImageDataGenerator(

    rescale=1./255,        

    horizontal_flip= False,

    rotation_range=90,

    vertical_flip = False,

    validation_split = 0.0

)
data_train_generator = augmentation_generate.flow(

    X_train,y_train,

    batch_size=32   # 5 data untuk proses propagasi maju



)
data_validation_gen = augmentation_generate.flow(

    X_test,y_test,

    batch_size=32     

)
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



model_cnn3 = Sequential()

shape=(28,28,1)

model_cnn3.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=shape))

model_cnn3.add(MaxPooling2D((2, 2)))

model_cnn3.add(Conv2D(64, kernel_size=(3,3), activation='relu'))

model_cnn3.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn3.add(Conv2D(128, kernel_size=(3,3), activation='relu'))

model_cnn3.add(MaxPooling2D(pool_size=(2, 2)))

model_cnn3.add(Flatten())                         # flatten array

model_cnn3.add(Dense(256, activation='relu'))     # Fungsi aktivasi relu berjumlah 256 buah

model_cnn3.add(Dense(10, activation='softmax'))    # Fungsi aktivasi softmax untuk 3 buah output (paper, rock, scissor)
import tensorflow as tf

loss_fn = tf.keras.losses.Poisson(reduction="auto", name="poisson")   # Loss function berbentuk fungsi Poisson digunakan untuk menghitung nilai error

model_cnn3.compile(loss=loss_fn, optimizer=tf.optimizers.Adam(), metrics=['accuracy']) # optimizer digunakan untuk update nilai hidden layer untuk updating nilai ke nilai yang lebih baik
model_cnn3.fit(

      data_train_generator, # Data train generator

      epochs=40,           # Jumlah epoch maksimal

      steps_per_epoch=65,  # Jumlah data yang diperlukan untuk menyelesaikan satu kali epoch

      validation_steps=5,  # Jumlah data validasi yang dilewatkan

      validation_data=data_validation_gen, verbose=1)
scores = model_cnn3.evaluate(X_test, y_test, verbose=0)

print("%s: %.2f%%" % (model_cnn3.metrics_names[1], scores[1]*100))

print("%s: %.2f%%" % (model_cnn3.metrics_names[0], scores[0]*100))