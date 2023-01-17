# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow

from scipy import stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
train_df.describe()



# Data is not normalised, 

# contained in a df, need to convert ot a numpy array for easier manipulation.
x_y = train_df.values

y = x_y[:,0]

x = x_y[:,1:]

x = x.reshape([42000,28,28,1])



y = tensorflow.keras.utils.to_categorical(

    y,

    num_classes=10

)

x = x / 255



print(y.shape)

print(x.shape)
def model():

    model = tensorflow.keras.Sequential()

    model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28,1)))

    model.add(tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tensorflow.keras.layers.Conv2D(128, (2, 2)))

    model.add(tensorflow.keras.layers.MaxPool2D(pool_size=(2, 2)))

    model.add(tensorflow.keras.layers.Flatten())

    model.add(tensorflow.keras.layers.Dense(512, activation="relu"))

    model.add(tensorflow.keras.layers.Dropout(0.7))

    model.add(tensorflow.keras.layers.Dense(10, activation='softmax'))

    return model



model = model()

model.summary()



model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,

              optimizer=tensorflow.keras.optimizers.Adam(),

              metrics=['accuracy'])



def schedule(index, lr):

    if index % 4 == 1:

        return lr/4

    else:

        return lr



LR_callback = tensorflow.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

best_model_callback = tensorflow.keras.callbacks.ModelCheckpoint("best_model", monitor='val_loss', save_best_only=True)
batch_size = 64



image_gen = tensorflow.keras.preprocessing.image.ImageDataGenerator(

    validation_split=0.1)

image_gen.fit(x)



train_generator = image_gen.flow(

    x,

    y=y,

    batch_size=batch_size,

    subset='training') # set as training data



validation_generator = image_gen.flow(

    x,

    y=y,

    batch_size=batch_size,

    subset='validation') # set as validation data



model.fit_generator(

    train_generator,

    validation_data = validation_generator,

    epochs = 10)
test_df = pd.read_csv("../input/test.csv")

x_test = test_df.values.reshape([28000,28,28,1])

y_test = np.argmax(model.predict(x_test), axis=1)

ids = np.arange(28000)

result = pd.DataFrame({"Label":y_test})

result.to_csv("result.csv", index_label="ImageId")