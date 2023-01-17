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

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from os import getcwd

import matplotlib.pyplot as plt

import matplotlib.cm as cm

train =  pd.read_csv("../input/digit-recognizer/train.csv")

test =  pd.read_csv("../input/digit-recognizer/test.csv")
test.shape
images = train.iloc[:,1:].values

images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)



print('images({0[0]},{0[1]})'.format(images.shape))





# convert from [0:255] => [0.0:1.0]

train_images = images.reshape(42000,28,28,1)
test_images = test.to_numpy()

print(test_images.shape)

test_images = test_images.reshape(28000,28,28,1)

test_images = test_images/255.0

print(test_images.shape)
len(test_images)
test_images.shape
df1 = train.iloc[:, 0:1]

train_labels = df1.to_numpy()

train_labels = train_labels.reshape(42000,)

train_labels.shape
train_labels = train_labels.astype('uint8')
model = tf.keras.models.Sequential([

      tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

      tf.keras.layers.MaxPooling2D(2, 2),

      tf.keras.layers.Flatten(),

      tf.keras.layers.Dense(128, activation='relu'),

      tf.keras.layers.Dense(10, activation='softmax')

    ])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(

        train_images, train_labels, epochs=12

    )
model.save("model")
model.predict(train_images[[0]])
x = np.argmax(model.predict(test_images[[4]]))

x
def display(img):

    

    # (784) => (28,28)

    one_image = img.reshape(28,28)

    

    plt.axis('off')

    plt.imshow(one_image, cmap=cm.binary)



# output image     

display(test_images[4])
sub = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

sub.head()
len(test_images)
len(sub)


for i in range(len(sub)):

    #temp  = sub.Label[i]

    img = test_images[[i]]

    pred = model.predict(img)

    pred = np.argmax(pred)

    sub.Label[i]  = pred

    

sub.to_csv("sample_submission.csv",index=False)

sub.head()