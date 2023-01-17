import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from keras import layers

from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Add

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout,GlobalMaxPooling2D, GlobalAveragePooling2D

from keras.models import Model, Sequential

from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

from keras.initializers import glorot_uniform

from keras.optimizers import RMSprop

import keras.backend as K

K.set_image_data_format("channels_last")

import time

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
label = np.array(train.label)

data = np.array(train.drop(["label"], axis=1))

 

# Normalization and one hot

X = data.reshape((42000, 28, 28, 1))

X = X / 255

enc = OneHotEncoder()

Y = enc.fit_transform(label.reshape((42000, 1))).toarray()

train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.1, random_state=0)

 

# Normalizetion

test_data = np.array(test)

test_data = test_data.reshape((28000, 28, 28, 1))

test_data = test_data / 255

 

# Take a look at the data at will

index = 25351

plt.imshow(data.reshape((42000, 28, 28))[index])

label[index]
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False, samplewise_std_normalization=False, zca_whitening=False, rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=False, vertical_flip=False)

datagen.fit(train_x)
model = [0] * 8

for j in range(0, 8):

    model[j] = Sequential()

    model[j].add(Conv2D(2 ** (j + 1), kernel_size=2, activation='relu', input_shape=(28,28,1)))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print(str(2 ** (j + 1)) + "个filters:")

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 6

for j in range(6):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=2 + j, activation='relu', input_shape=(28,28,1)))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("过滤器大小：" + str(2 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("---------------------------------------------------------------------------------------------")

print("same：")

start = time.time()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=20, steps_per_epoch= len(train_x) // 64)

end = time.time()

print(str(end - start) + "秒")

print("---------------------------------------------------------------------------------------------")

 

model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding="valid", activation='relu', input_shape=(28,28,1)))

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("---------------------------------------------------------------------------------------------")

print("same：")

start = time.time()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

end = time.time()

print(str(end - start) + "秒")

print("---------------------------------------------------------------------------------------------")
model = [0] * 5

for j in range(5):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    

    if j > 0:

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    if j > 1:

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    if j > 2:

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    if j > 3:

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("卷积层层数：" + str(1 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 5

for j in range(5):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=j+2))

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("池化大小：" + str(2 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=3, padding="same"))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=3))

 

model.add(Flatten())

model.add(Dense(256, activation="relu"))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("---------------------------------------------------------------------------------------------")

print("最后两层池化：")

start = time.time()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

end = time.time()

print(str(end - start) + "秒")

print("---------------------------------------------------------------------------------------------")
model = [0] * 2

for j in range(2):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    if j > 0:

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

        model[j].add(MaxPooling2D(pool_size=3, padding="same"))

        model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

        model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("模块层数：" + str(1 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 2

for j in range(2):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Conv2D(32 * 2 ** (j + 1), kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32 * 2 ** (j + 1), kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32 * 2 ** (j + 1), kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32 * 2 ** (j + 1), kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32 * 2 ** j, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("新模块filters数：" + str(32 * 2 ** (j + 1)))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 6

for j in range(6):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Conv2D(64, kernel_size=2 + j, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=2 + j, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=2 + j, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=2 + j, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(64, kernel_size=2 + j, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("新模块过滤器大小：" + str(2 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 5

for j in range(5):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=2 + j, padding="same"))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=2 + j, padding="same"))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("新模块池化大小：" + str(2 + j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 7

for j in range(7):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    if j == 0 or j == 3 or j == 4 or j == 6:

        model[j].add(Dropout(0.1))

    

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    if j == 1 or j == 3 or j == 5 or j == 6:

        model[j].add(Dropout(0.1))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    

    if j == 2 or j == 4 or j == 5 or j == 6:

        model[j].add(Dropout(0.1))

    

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("添加dropout的方式：" + str(j))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 7

for j in range(7):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Dropout(0.1 * (j + 1)))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    

    model[j].add(Dropout(0.1))

    

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("第一个dropout的参数：" + str(0.1 * (j + 1)))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = [0] * 6

for j in range(6):

    model[j] = Sequential()

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(32, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    model[j].add(Conv2D(64, kernel_size=5, padding="same", activation='relu', input_shape=(28,28,1)))

    model[j].add(MaxPooling2D(pool_size=3, padding="same"))

    

    model[j].add(Dropout(0.4))

    

    model[j].add(Flatten())

    model[j].add(Dense(256, activation="relu"))

    

    model[j].add(Dropout(0.2 + j * 0.1))

    

    model[j].add(Dense(10, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    print("---------------------------------------------------------------------------------------------")

    print("第二个dropout的参数：" + str(0.1 * (j + 2)))

    start = time.time()

    model[j].fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

    end = time.time()

    print(str(end - start) + "秒")

    print("---------------------------------------------------------------------------------------------")
model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0), input_shape=(28,28,1)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

    

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

    

model.add(Dropout(0.4))

    

model.add(Flatten())

model.add(Dense(256, activation="relu", kernel_initializer=glorot_uniform(seed=0)))

    

model.add(Dropout(0.5))

    

model.add(Dense(10, activation='softmax', kernel_initializer=glorot_uniform(seed=0)))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

print("---------------------------------------------------------------------------------------------")

print("加BN：")

start = time.time()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

end = time.time()

print(str(end - start) + "秒")

print("---------------------------------------------------------------------------------------------")
model = Sequential()

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0), input_shape=(28,28,1)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

model.add(Conv2D(32, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

    

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

model.add(Conv2D(64, kernel_size=5, padding="same", activation='relu', kernel_initializer=glorot_uniform(seed=0)))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=3, padding="same"))

    

model.add(Dropout(0.4))

    

model.add(Flatten())

model.add(Dense(256, activation="relu", kernel_initializer=glorot_uniform(seed=0)))

    

model.add(Dropout(0.5))

    

model.add(Dense(10, activation='softmax', kernel_initializer=glorot_uniform(seed=0)))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("---------------------------------------------------------------------------------------------")

start = time.time()

model.fit_generator(datagen.flow(train_x, train_y, batch_size=64), validation_data=(val_x, val_y), epochs=1, steps_per_epoch= len(train_x) // 64)

end = time.time()

print(str(end - start) + "秒")

print("---------------------------------------------------------------------------------------------")
predict = np.argmax(model.predict(test_data), axis=1)

 

sample_submission["Label"] = predict

sample_submission.to_csv("1.csv", index=False)
model.save_weights("my_model_weights_614_1.h5")

 

model.load_weights("my_model_weights_614_1.h5")
model.load_weights("my_model_weights_600_2.h5")

predict_600_2 = np.argmax(model.predict(test_data), axis=1)

 

model.load_weights("my_model_weights_628_2.h5")

predict_628_2 = np.argmax(model.predict(test_data), axis=1)

 

model.load_weights("my_model_weights_642_2.h5")

predict_642_2 = np.argmax(model.predict(test_data), axis=1)
def combine_model(predict1, predict2, predict3):

    not_equal_index = np.unique(np.hstack((np.where(predict1 != predict2)[0], np.where(predict1 != predict3)[0], np.where(predict2 != predict3)[0])))

    predict = np.copy(predict1)

    for i in not_equal_index:

        if (predict2[i] == predict3[i]) and (predict2[i] != predict1[i]):

            predict[i] = predict2[i]

    return predict
sample_submission["Label"] = combine_model(predict_642_2, predict_600_2, predict_628_2)

sample_submission.to_csv("1.csv", index=False)