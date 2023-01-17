# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



digit_file = "../input/train.csv"

digit_data = np.loadtxt(digit_file, skiprows=1, delimiter=',')

x, y = prep_data(digit_data)







from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



batch_size = 16

#Model Arch (Conv2D-->Conv2D--->Flatten--->Dense--->Dense)

digit_model = Sequential()

digit_model.add(Conv2D(16, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

digit_model.add(Conv2D(16, (3, 3), activation='relu'))

digit_model.add(Flatten())

digit_model.add(Dense(128, activation='relu'))

digit_model.add(Dense(num_classes, activation='softmax'))



digit_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



digit_model.fit(x, y,

          batch_size=batch_size,

          epochs=3,

          validation_split = 0.2)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout



batch_size = 16



digit_model = Sequential()

digit_model.add(Conv2D(16, kernel_size=(3, 3),

                 activation='relu',

                 input_shape=(img_rows, img_cols, 1)))

digit_model.add(Conv2D(16, (3, 3), activation='relu', strides=2))

digit_model.add(Flatten())

digit_model.add(Dense(128, activation='relu'))

digit_model.add(Dense(num_classes, activation='softmax'))



digit_model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer='adam',

              metrics=['accuracy'])



digit_model.fit(x, y,

          batch_size=batch_size,

          epochs=10,

          validation_split = 0.2)
results=digit_model.predict(x)



results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False,quoting=csv.QUOTENONNUMERIC)
