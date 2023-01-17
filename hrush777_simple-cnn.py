# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/train.csv')

data.head()
data.shape
img_data = data.iloc[:,1:].values

lables = data['label']
img_data = np.array(img_data)

img_data.shape
img_data = img_data.reshape(-1,28,28,1)
img_data.shape
plt.imshow(img_data[1].reshape(28,28))
from keras.utils import to_categorical

Y = to_categorical(lables)
Y.shape
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.fit(img_data, Y,

          batch_size=128,

          epochs=20,

          verbose=1)
submission = pd.read_csv('../input/sample_submission.csv')

submission.shape

submission.head()
img_id = submission['ImageId']

img_id = np.array(img_id)
test_data = pd.read_csv('../input/test.csv')

test_data.head()
test_data.shape
test_data = np.array(test_data).reshape(-1,28,28,1)

test_data.shape
pred = model.predict_classes(test_data)
pred.shape
pred
plt.imshow(test_data[0].reshape(28,28))
img_id = img_id.reshape(-1,1)

pred = pred.reshape(-1,1)
output = np.array(np.concatenate((img_id, pred), 1))
output = pd.DataFrame(output,columns = ["ImageId","Label"])
output.head()
output.to_csv('out.csv',index = False)