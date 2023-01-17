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

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train.shape
train_data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test_data=pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
submission=pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train_data.head()
train_data.isnull().sum()
import seaborn as sns
sns.countplot(train_data['label'])

X_train=train_data.drop('label',axis=1).to_numpy()/255
y_train=train_data['label']
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=2)

import matplotlib.pyplot as plt
from numpy.random import randint
plt.imshow(x_train[randint(1,100)].reshape(28,28))

y_train[1]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10,activation='softmax')
])
model.summary()
x_train=x_train.reshape(-1,28,28,1)
y_train.shape
from keras.preprocessing.image import ImageDataGenerator
agu=tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
)
agu.fit(x_train)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3,
                                                    verbose=1,
                                                    min_lr=0.000001,
                                                   )

model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.fit(agu.flow(x_train, y_train, batch_size=64),
          steps_per_epoch=len(x_train) / 64, epochs=30,
         validation_data = (x_test.reshape(-1,28,28,1),y_test),
          callbacks=[lr_scheduler],
         )
test_data=test_data.to_numpy()/255.0

prediction=model.predict_classes(test_data.reshape(-1,28,28,1))
prediction[:5]
submission['Label'] = prediction
submission.to_csv("submission.csv" , index = False)
submission.head()