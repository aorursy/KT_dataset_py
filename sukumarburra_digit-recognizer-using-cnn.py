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
import pandas as pd

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout





img_rows, img_cols = 28, 28

num_classes = 10



def data_prep(raw):

    out_y = to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



train_file = "../input/digit-recognizer/train.csv"

raw_data = pd.read_csv(train_file)



x, y = data_prep(raw_data)



model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3),

                 strides=2,

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

#model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), strides=2, activation='relu'))

#model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])

model.fit(x, y,

          batch_size=128,

          epochs=4,

          validation_split = 0.2)
# predict results

test_file = "../input/digit-recognizer/test.csv"

test = pd.read_csv(test_file)

test = test/255

test = test.values.reshape(-1,img_rows, img_cols, 1)

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_3.csv",index=False)