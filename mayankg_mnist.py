# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.utils import shuffle



train_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv', header=0)

test_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv', header=0)
print(train_data.shape)

print(test_data.shape)

print(len(train_data))
x = range(len(train_data))

train_images = []

train_labels = []



train_data = shuffle(train_data)



for i in x:

    temp_images = np.array_split(train_data.iloc[i,1:], 28)

    temp_labels = train_data.iloc[i,0]

    train_images.append(temp_images)

    train_labels.append(temp_labels)

train_images = np.array(train_images).astype('float')

train_labels = np.array(train_labels).astype('float')
train_images = np.expand_dims(train_images, axis = 3)

train_images = train_images / 255.
test_images = []

for i in range(len(test_data)):

    temp_image = np.array_split(test_data.iloc[i,:], 28)

    test_images.append(temp_image)

test_images = np.array(test_images).astype('float')

test_images = np.expand_dims(test_images, axis = 3)

test_images = test_images / 255.
print(train_images.shape)

print(train_labels.shape)

print(test_images.shape)
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(16,(3,3), activation = 'relu', input_shape = (28,28,1)),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Conv2D(64,(3,3), activation = 'relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])



model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'rmsprop', metrics = ['acc'])
model.summary()
history = model.fit(

    x = train_images,

    y = train_labels,

    batch_size = 32,

    epochs = 25,

    shuffle = True,

    verbose = 2,

    validation_split = 0.2,

)
predictions = model.predict(test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": np.argmax(predictions, axis=1)})

submissions.to_csv("my_submissions.csv", index=False, header=True)