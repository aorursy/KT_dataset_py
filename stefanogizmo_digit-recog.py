# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#digit_file_train_path = '../input/train.csv'

#digit_file_test_path = '../input/test.csv'

#digit_data_train = pd.read_csv(digit_file_train_path)

#digit_data_test=pd.read_csv(digit_file_test_path)



num_classes = 10

img_rows, img_cols = 28, 28

def data_prep(raw):

    out_y = keras.utils.to_categorical(raw.label, num_classes)



    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y



def data_prep_test(raw):

    num_images = raw.shape[0]

    x_as_array = raw.values[:,:]

    x_shaped_array = x_as_array.reshape(num_images, img_rows, img_cols, 1)

    out_x = x_shaped_array / 255

    return out_x

train_file = "../input/train.csv"

raw_data = pd.read_csv(train_file)

x, y = data_prep(raw_data)



test_file ="../input/test.csv"

raw_data_test =pd.read_csv(test_file)

x_test = data_prep_test(raw_data_test)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



model.fit(x, y,

          batch_size=256,

          epochs=5,

          validation_split = 0.25)









predictions = model.predict_classes(x_test, verbose=0)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

                         "Label": predictions})

submissions.to_csv("Submission.csv", index=False, header=True)


print(x_test)