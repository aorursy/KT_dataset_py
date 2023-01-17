# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import keras
from keras.layers import Conv2D , MaxPooling2D , Dense , Flatten , Dropout
from keras.optimizers import Adam
from keras.models import Sequential

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
train_df.head()
test_df.head()
train_data = np.array(train_df, dtype='float32')
test_data = np.array(test_df, dtype='float32')

x_train = train_data[:, 1:] / 255
y_train = train_data[:, 0]

x_test = test_data[:,:] / 255
x_train, x_validate, y_train, y_validate = train_test_split(
    x_train, y_train, test_size=0.2, random_state=12345,
)
import matplotlib.pyplot as plt
image = x_train[50, :].reshape((28, 28))

plt.imshow(image)
plt.show()
im_rows = 28
im_cols = 28
batch_size = 512
im_shape = (im_rows, im_cols, 1)
x_train.shape , x_test.shape
x_train = x_train.reshape(x_train.shape[0], *im_shape)
x_test = x_test.reshape(28000, 28, 28, 1)
x_validate = x_validate.reshape(x_validate.shape[0], *im_shape)

cnn_model = Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    
    Flatten(),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(lr=0.001),
    metrics=['accuracy']
)

cnn_model.fit(
    x_train, y_train, batch_size=batch_size,
    epochs=10, verbose=1,
    validation_data=(x_validate, y_validate)
)
pred = cnn_model.predict(x_test)
predictions = []
for arr in pred:
    for i in arr:
        if i==max(arr):
            predictions.append(list(arr).index(i))
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
# Generate csv file
submissions.to_csv("submission.csv", index=False, header=True)