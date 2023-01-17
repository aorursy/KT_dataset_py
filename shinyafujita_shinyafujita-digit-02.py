import os
import csv
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dropout
df_train_tmp = pd.read_csv("../input/train.csv")
df_test_tmp = pd.read_csv("../input/test.csv")
def show(image):
    pylab.gray()
    pylab.imshow(image.reshape(28, 28))
tmp_data = df_train_tmp.iloc[:,1:].values.reshape(len(df_train_tmp), 28, 28, 1).astype("float32") / 255
tmp_label = keras.utils.to_categorical(df_train_tmp["label"].values, 10)

train_data = tmp_data[:30000]
train_label = tmp_label[:30000]
valid_data = tmp_data[30000:]
valid_label = tmp_label[30000:]

test_data = df_test_tmp.values.reshape(len(df_test_tmp), 28, 28, 1).astype("float32") / 255
i = 40
show(train_data[i])
print(np.argmax(train_label[i]))
datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=math.pi/4,
        zoom_range=0.3,
        fill_mode="constant",
        cval=-1,
    )
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=["accuracy"])
BATCH_SIZE = 1024
history = model.fit_generator(
    datagen.flow(train_data, train_label, batch_size=BATCH_SIZE),
    steps_per_epoch=len(train_data)/BATCH_SIZE, epochs=100,
    validation_data=(valid_data, valid_label)
).history
#Accuracy
plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
#loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
test_predicted = np.argmax(model.predict(test_data), axis=1)
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["ImageId", "Label"])
    for i in range(len(test_predicted)):
        writer.writerow([i + 1, test_predicted[i]])
i = 60
show(test_data[i])
test_predicted[i]
