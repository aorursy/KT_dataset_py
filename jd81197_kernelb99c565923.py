# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from IPython.display import HTML



import base64



from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau, CSVLogger

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")

test=pd.read_csv("../input/test.csv")
y=train["label"]

x=train.drop(labels=["label"], axis=1)
x.shape
x = x.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)
y = to_categorical(y, num_classes=10)
x_training, x_validation, y_training, y_validation = train_test_split(x,

                                                                      y,

                                                                      test_size=0.1,

shuffle=True)
data_generator = ImageDataGenerator(rescale=1./255,

                                    rotation_range=10,

                                    zoom_range=0.15, 

                                    width_shift_range=0.1,

height_shift_range=0.1)
data_generator.fit(x_training)
model = Sequential()



model.add(Conv2D(filters=32,

                 kernel_size=(5,5),

                 padding='Same', 

                 activation='relu',

                 input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32,

                 kernel_size=(5,5),

                 padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.5))



model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 

                 activation='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(8192, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(2048, activation='relu'))

model.add(Dropout(0.5))



model.add(Dense(10, activation="softmax"))



model.compile(optimizer=RMSprop(lr=0.0001,

                                rho=0.9,

                                epsilon=1e-08,

                                decay=0.00001),

              loss="categorical_crossentropy",

metrics=["accuracy"])
TRAINING_LOGS_FILE = "training_logs.csv"

history = model.fit_generator(data_generator.flow(x_training,

                                                  y_training,

                                                  batch_size=512),

                              epochs=100,

                              validation_data=(x_validation, y_validation),

                              verbose=1,

                              steps_per_epoch=x_training.shape[0] // 512,

                             )
MODEL_FILE = "model.h5"
model.save_weights(MODEL_FILE)

predictions = model.predict_classes(test, verbose=1)
KAGGLE_SUBMISSION_FILE="submission_digit.csv"

pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),

              "Label":predictions}).to_csv(KAGGLE_SUBMISSION_FILE,

                                           index=False,

                                           header=True)
print(os.listdir("./"))
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)