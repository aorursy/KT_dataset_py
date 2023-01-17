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
train_datagen = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_datagen = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
from tensorflow import keras
from tensorflow.keras import Input, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, SeparableConv2D
from tensorflow.keras.layers import BatchNormalization, Dropout, Activation, add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

train_datagen.shape, test_datagen.shape
train_y = train_datagen["label"]
train_y = keras.utils.to_categorical(train_y, num_classes = 10)

train_x = train_datagen.drop("label", axis=1)
train_x = train_x.values.reshape((-1, 28, 28, 1))
test_x = test_datagen.values.reshape((-1, 28, 28, 1))
train_x.shape, test_x.shape, train_y.shape
import matplotlib.pyplot as plt
plt.figure(0, figsize=(4 ,4))
plt.imshow(np.reshape(train_x[300], (28, 28)));
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def max_unit(args):
    inputs, depthconv_output = args
    return tf.maximum(inputs, depthconv_output)

def FReLU(inputs, kernel_size = 3):
    x = DepthwiseConv2D(kernel_size, strides=(1, 1), padding="same")(inputs)
    x = BatchNormalization()(x)
    
    x_shape = K.int_shape(x)
    
    x = Lambda(max_unit, output_shape=(x_shape[1], x_shape[2], x_shape[3]))([inputs, x])
    return x
# input_tensor = Input(shape=(28, 28, 1), name="input")
# x = SeparableConv2D(64, kernel_size=3, strides=1, activation="relu", name="conv_1")(input_tensor)
# x = BatchNormalization(name="batch_1")(x)
# x = SeparableConv2D(64, kernel_size=3, strides=1, activation="relu", name="conv_2")(x)
# x = BatchNormalization(name="batch_2")(x)
# x = SeparableConv2D(64, kernel_size=5, strides=2, activation="relu", padding="same", name="conv_3")(x)
# x = BatchNormalization(name="batch_3")(x)
# x = Dropout(0.45, name="dropout_1")(x)

# x = SeparableConv2D(128, kernel_size=3, strides=1, activation="relu", name="conv_4")(x)
# x = BatchNormalization(name="batch_4")(x)
# x = SeparableConv2D(128, kernel_size=3, strides=1, activation="relu", name="conv_5")(x)
# x = BatchNormalization(name="batch_5")(x)
# x = SeparableConv2D(128, kernel_size=5, strides=2, activation="relu", padding="same", name="conv_6")(x)
# x = BatchNormalization(name="batch_6")(x)
# x = Dropout(0.45, name="dropout_2")(x)

# x = SeparableConv2D(256, kernel_size=3, strides=1, activation="relu", name="conv_7")(x)
# x = BatchNormalization(name="batch_7")(x)
# x = SeparableConv2D(256, kernel_size=5, strides=2, activation="relu", padding="same", name="conv_8")(x)
# x = BatchNormalization(name="batch_8")(x)
# x = Flatten(name="flatten")(x)
# x = Dense(128, activation="relu", name="dense_1")(x)
# x = Dropout(0.45, name="dropout_3")(x)
# output_tensor = Dense(10, activation="softmax", name="output")(x)

# model = Model(input_tensor, output_tensor)
from hyperas import optim 
from hyperas.distributions import choice, uniform
from hyperopt import Trials
from hyperopt import STATUS_OK
from hyperopt import tpe
# input_tensor = Input(shape=(28, 28, 1), name="input")
# x = SeparableConv2D(64, kernel_size=3, strides=1, name="conv_1")(input_tensor)
# x = BatchNormalization(name="batch_1")(x)
# x = FReLU(x)
# x = SeparableConv2D(64, kernel_size=3, strides=1, name="conv_2")(x)
# x = BatchNormalization(name="batch_2")(x)
# x = FReLU(x)
# x = SeparableConv2D(64, kernel_size=5, strides=2, padding="same", name="conv_3")(x)
# x = BatchNormalization(name="batch_3")(x)
# x = FReLU(x)
# x = Dropout(0.45, name="dropout_1")(x)

# x = SeparableConv2D(128, kernel_size=3, strides=1, name="conv_4")(x)
# x = BatchNormalization(name="batch_4")(x)
# x = FReLU(x)
# x = SeparableConv2D(128, kernel_size=3, strides=1, name="conv_5")(x)
# x = BatchNormalization(name="batch_5")(x)
# x = FReLU(x)
# x = SeparableConv2D(128, kernel_size=5, strides=2, padding="same", name="conv_6")(x)
# x = BatchNormalization(name="batch_6")(x)
# x = FReLU(x)
# x = Dropout(0.45, name="dropout_2")(x)

# x = SeparableConv2D(256, kernel_size=3, strides=1, name="conv_7")(x)
# x = BatchNormalization(name="batch_7")(x)
# x = FReLU(x)
# x = SeparableConv2D(256, kernel_size=5, strides=2, padding="same", name="conv_8")(x)
# x = BatchNormalization(name="batch_8")(x)
# x = FReLU(x)
# x = Flatten(name="flatten")(x)
# x = Dense(128, activation="relu", name="dense_1")(x)
# x = Dropout(0.45, name="dropout_3")(x)
# output_tensor = Dense(10, activation="softmax", name="output")(x)

# model = Model(input_tensor, output_tensor)
model.summary()
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)
X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape, y_test.shape
def compile_model(model):
    model.compile(loss="categorical_crossentropy",
                 optimizer="rmsprop",
                 metrics=["acc"])
def fit_model(model):
    EPOCHS=100

    callbacks_list = [EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30), 
                      ModelCheckpoint(filepath="model.h5", monitor="val_loss", save_bast_only=True),
                      ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=10)]

    history = model.fit(X_train, 
                        y_train, 
                        epochs=EPOCHS, 
                        batch_size=128,
                        callbacks=callbacks_list, 
                        validation_data=(X_val, y_val))
def create_model(X_train, X_test, y_train, y_test):
    input_tensor = Input(shape=(28, 28, 1), name="input")
    x = SeparableConv2D({{choice([64, 128])}}, kernel_size=3, strides=1, name="conv_1")(input_tensor)
    x = BatchNormalization(name="batch_1")(x)
    x = FReLU(x)
    x = SeparableConv2D({{choice([64, 128])}}, kernel_size=3, strides=1, name="conv_2")(x)
    x = BatchNormalization(name="batch_2")(x)
    x = FReLU(x)
    x = SeparableConv2D({{choice([64, 128])}}, kernel_size=5, strides=2, padding="same", name="conv_3")(x)
    x = BatchNormalization(name="batch_3")(x)
    x = FReLU(x)
    x = Dropout({{uniform(0, 1)}}, name="dropout_1")(x)

    x = SeparableConv2D({{choice([128, 256])}}, kernel_size=3, strides=1, name="conv_4")(x)
    x = BatchNormalization(name="batch_4")(x)
    x = FReLU(x)
    x = SeparableConv2D({{choice([128, 256])}}, kernel_size=3, strides=1, name="conv_5")(x)
    x = BatchNormalization(name="batch_5")(x)
    x = FReLU(x)
    x = SeparableConv2D({{choice([128, 256])}}, kernel_size=5, strides=2, padding="same", name="conv_6")(x)
    x = BatchNormalization(name="batch_6")(x)
    x = FReLU(x)
    x = Dropout({{uniform(0, 1)}}, name="dropout_2")(x)

    x = SeparableConv2D({{choice([256, 512])}}, kernel_size=3, strides=1, name="conv_7")(x)
    x = BatchNormalization(name="batch_7")(x)
    x = FReLU(x)
    x = SeparableConv2D({{choice([256, 512])}}, kernel_size=5, strides=2, padding="same", name="conv_8")(x)
    x = BatchNormalization(name="batch_8")(x)
    x = FReLU(x)
    x = Flatten(name="flatten")(x)
    x = Dense({{choice([64, 128])}}, activation="relu", name="dense_1")(x)
    x = Dropout({{uniform(0, 1)}}, name="dropout_3")(x)
    output_tensor = Dense(10, activation="softmax", name="output")(x)

    model = Model(input_tensor, output_tensor)
    
    compile_model(model)
    
    fit_model(model)
    
    validation_acc = np.amax(history.history["val_acc"])
    print("Best validation accuracy of epoch: ", validation_acc)
    return {"loss": -valdation_acc, "status": STATUS_OK, "model": model}
def data(X_train, X_test, y_train, y_test):
    return X_train, X_test, y_train, y_test
if __name__ == "__main__":
    best_run, best_model = optim.minimize(
        model=create_model,
        data=data,
        algo=tpe.suggest,
        max_evals=5,
        trials=Trials(),
        notebook_name="__notebook_source__")
    
    print("Evaluation of best performing model:")
    print(best_model.evaluate(y_train, y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
model.evaluate(X_test, y_test)
plt.plot(history.history["val_acc"])
plt.plot(history.history["acc"])
plt.title("accuracy score")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(["train", "test"], loc="upper left")
plt.show()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("loss score")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(["train", "test"])
plt.show()
image_id = list(test_datagen.index)
image_id = [i+1 for i in image_id]

prediction = model.predict(test_x)
prediction = np.argmax(prediction, axis=1)

data = {"ImageId": image_id, "Label": prediction}
results = pd.DataFrame(data)
results.to_csv("submission_7.csv", index=False)