# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test_data = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

val_data = pd.read_csv("/kaggle/input/Kannada-MNIST/Dig-MNIST.csv")
train_data.shape, val_data.shape, test_data.shape
train_data.head()
test_data.head()
label_train = train_data["label"]

train_data = train_data.drop("label", axis= 1)
label_val = val_data["label"]

val_data = val_data.drop("label", axis=1)
test_data = test_data.drop("id", axis=1)
train_data.shape, val_data.shape, test_data.shape
train_data = train_data / 255.0

val_data = val_data / 255.0

test_data = test_data / 255.0
train_data = train_data.values.reshape(-1,28,28,1)

val_data = val_data.values.reshape(-1,28,28,1)

test_data = test_data.values.reshape(-1,28,28,1)
train_data.shape, val_data.shape, test_data.shape
def build_model():

    model = tf.keras.models.Sequential()

    

    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), activation = "relu",

                                    input_shape = (28,28,1), padding = "same"))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), activation = "relu", padding = "same"))

    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Dropout(0.25))



    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", padding = "same"))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = "relu", padding = "same"))

    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Dropout(0.25))

    

    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", padding = "same"))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (3,3), activation = "relu", padding = "same"))

    #model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.MaxPool2D(pool_size = (2,2)))

    model.add(tf.keras.layers.Dropout(0.25))

    

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units = 256, activation = "relu"))

    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Dense(units = 10, activation = "softmax"))

              

    return model

label_train = tf.keras.utils.to_categorical(label_train, num_classes=10)

label_val = tf.keras.utils.to_categorical(label_val, num_classes= 10)
model = build_model()
model.summary()
model.compile(optimizer= "adam", loss= "categorical_crossentropy", metrics= ["accuracy"])
learn_rate_red = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", verbose=1, patience=3, factor= 0.3,

                                                     min_lr= 0.00001)
history = model.fit(train_data, label_train, validation_data= (val_data, label_val), batch_size= 64, 

          epochs= 3, callbacks= [learn_rate_red])
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], color='b')

plt.plot(history.history['val_loss'], color='r')

plt.show()

plt.plot(history.history['accuracy'], color='b')

plt.plot(history.history['val_accuracy'], color='r')

plt.show()
pred = model.predict(test_data)
pred.shape
pred = np.argmax(pred, axis=1)
pred_df = pd.DataFrame(data= pred)
pred_df.head()
pred_df = pred_df.reset_index()
pred_df.columns = ["id", "label"]
pred_df.head()
pred_df.to_csv("submission.csv", index= False)