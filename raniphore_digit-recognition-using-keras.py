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
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

train_df.info()
test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_df.info()
train_df.head()
test_df.head()
train_label = train_df.pop("label")

train_img = train_df

test_img = test_df.copy()
import matplotlib.pyplot as plt

plt.imshow(train_img.iloc[0,:].values.reshape(28,28),cmap='gray')

plt.show()
print(train_img.shape)
train_images = train_img.values.reshape((42000, 28, 28, 1))

train_images = train_images.astype('float32') / 255



test_images = test_img.values.reshape((28000, 28, 28, 1))

test_images = test_images.astype('float32') / 255
from keras.utils import to_categorical

train_labels = to_categorical(train_label)
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, batch_size=64)
y_pred = model.predict(test_images)

y_pred.shape
pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv').head()
out = [np.argmax(i) for i in y_pred]
len(out)
out_serial = range(1,len(out)+1)

len(out_serial)
submission_df = pd.DataFrame(zip(out_serial,out),columns=['ImageId','Label'])

submission_df.head()
submission_df.to_csv('submission.csv',index=False, header=True)