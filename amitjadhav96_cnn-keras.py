import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ReduceLROnPlateau


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Load the data
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")

train.head()

test.head()
x_train = train.drop(labels = ["label"],axis = 1) 
y_train = train["label"]

# Normalize data set to 0-to-1 range
x_train = x_train.astype('float32')
test = test.astype('float32')
x_train /= 255
test /= 255
y_train = keras.utils.to_categorical(y_train, 10)

test.values
x_train
### Reshaping
x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)
dataAug = ImageDataGenerator(
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False)  # randomly flip images

dataAug.fit(x_train)
### Splitting into training and validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state=42)
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.000001)
### Designing the model
model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=(28,28,1), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
model.add(Dropout(0.15))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3),activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),strides = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation="relu"))

model.add(Dense(10, activation="softmax"))

model.compile(
    loss='categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)
model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=30,
    validation_data=(x_val, y_val),
    shuffle=True,
    callbacks = [lr_reduce]
)

model.summary()
plot_model(model, to_file='Model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))
results = model.predict(test)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("submission.csv",index=False)
submission
