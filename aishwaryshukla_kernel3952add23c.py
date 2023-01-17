import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.losses import CategoricalCrossentropy

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import Accuracy

from tensorflow.keras.callbacks import ReduceLROnPlateau
train_df = pd.read_csv("../input/Kannada-MNIST/train.csv")

dig_mnist = pd.read_csv("../input/Kannada-MNIST/Dig-MNIST.csv")

test_df = pd.read_csv("../input/Kannada-MNIST/test.csv")
train_df.head()
dig_mnist.head()
test_df.head()
y_train = train_df["label"].values

y_val = dig_mnist["label"].values
x_train = train_df[train_df.keys().drop(["label"])].values

x_val = dig_mnist[dig_mnist.keys().drop(["label"])].values

x_test = test_df[test_df.keys().drop(["id"])].values
x_test.shape
x_train = x_train.reshape(-1,28,28,1)

x_val = x_val.reshape(-1, 28, 28, 1)

x_test = x_test.reshape(-1, 28,28, 1)
x_train = x_train/255

x_val = x_val/255

x_test = x_test/255
x_train.shape, x_val.shape, x_test.shape
y_train = to_categorical(y_train)

y_val = to_categorical(y_val)
y_train.shape, y_val.shape
plt.imshow(x_val[6][:,:,0])
plt.imshow(x_train[6][:,:,0])
def create_model():

    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size=(3,3), padding="same", activation="relu", input_shape=(28,28,1)))

    model.add(Conv2D(filters = 16, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(filters = 32, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(Conv2D(filters = 32, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(filters = 64, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(Conv2D(filters = 64, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(2,2))

    

    model.add(Conv2D(filters = 128, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(Conv2D(filters = 128, kernel_size=(3,3), padding="same", activation="relu"))

    model.add(MaxPooling2D(2,2))

    model.add(Flatten())

    model.add(Dense(units=256, activation="relu"))

    model.add(Dropout(0.1))

    model.add(Dense(units=64, activation="relu"))

    model.add(Dense(units=10, activation="softmax"))

    return model
model = create_model()

model.compile(optimizer=Adam(lr=0.0001), loss=CategoricalCrossentropy(), metrics=["accuracy"])
model.summary()
datagen = ImageDataGenerator(

    samplewise_center=True,

    samplewise_std_normalization=True,

    rotation_range=30,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.2

)
datagen.fit(x_train)
lr = ReduceLROnPlateau(min_lr=0.00001, patience=4, verbose=1, monitor="loss")
BATCH_SIZE=64
history = model.fit(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE), epochs=20, 

                    steps_per_epoch = x_train.shape[0]//BATCH_SIZE, validation_data = (x_val, y_val), 

                    callbacks=[lr], verbose = 1)
preds = model.predict(x_test)
preds = np.argmax(preds, axis=1)
sam = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")
sam["label"] = preds
sam.to_csv("submission.csv", index=False)