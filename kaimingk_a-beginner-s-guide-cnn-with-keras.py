import gc

import random as rd

import time

from math import pi



import keras

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sbn

import tensorflow as tf

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback

from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,

                          MaxPool2D, ReLU)

from PIL import Image

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split

%matplotlib inline
print("Loading...")

data_train = pd.read_csv("../input/train.csv")

data_test = pd.read_csv("../input/test.csv")

print("Done!")
print("Training data: {} rows, {} columns.".format(data_train.shape[0], data_train.shape[1]))

print("Test data: {} rows, {} columns.".format(data_test.shape[0], data_test.shape[1]))
x_train = data_train.values[:, 1:]

y_train = data_train.values[:, 0]
def convert_2d(x):

    """x: 2d numpy array. m*n data image.

       return a 3d image data. m * height * width * channel."""

    if len(x.shape) == 1:

        m = 1

        height = width = int(np.sqrt(x.shape[0]))

    else:

        m = x.shape[0]

        height = width = int(np.sqrt(x.shape[1]))



    x_2d = np.reshape(x, (m, height, width, 1))

    

    return x_2d
x_display = convert_2d(data_train.values[0, 1:])

plt.imshow(x_display.squeeze(), cmap="gray")
def crop_image(x, y, min_scale):

    """x: 2d(m*n) numpy array. 1-dimension image data;

       y: 1d numpy array. The ground truth label;

       min_scale: float. The minimum scale for cropping.

       return zoomed images.

       This function crops the image, enlarges the cropped part and uses it as augmented data."""

    # convert the data to 2-d image. images should be a m*h*w*c numpy array.

    images = convert_2d(x)

    # m is the number of images. Since this is a gray-scale image scale from 0 to 255, it only has one channel.

    m, height, width, channel = images.shape

    

    # tf tensor for original images

    img_tensor = tf.placeholder(tf.int32, [1, height, width, channel])

    # tf tensor for 4 coordinates for corners of the cropped image

    box_tensor = tf.placeholder(tf.float32, [1, 4])

    box_idx = [0]

    crop_size = np.array([height, width])

    # crop and resize the image tensor

    cropped_img_tensor = tf.image.crop_and_resize(img_tensor, box_tensor, box_idx, crop_size)

    # numpy array for the cropped image

    cropped_img = np.zeros((m, height, width, 1))



    with tf.Session() as sess:



        for i in range(m):

            

            # randomly select a scale between [min_scale, min(min_scale + 0.05, 1)]

            rand_scale = np.random.randint(min_scale * 100, np.minimum(min_scale * 100 + 5, 100)) / 100

            # calculate the 4 coordinates

            x1 = y1 = 0.5 - 0.5 * rand_scale

            x2 = y2 = 0.5 + 0.5 * rand_scale

            # lay down the cropping area

            box = np.reshape(np.array([y1, x1, y2, x2]), (1, 4))

            # save the cropped image

            cropped_img[i:i + 1, :, :, :] = sess.run(cropped_img_tensor, feed_dict={img_tensor: images[i:i + 1], box_tensor: box})

    

    # flat the 2d image

    cropped_img = np.reshape(cropped_img, (m, -1))

    cropped_img = np.concatenate((y.reshape((-1, 1)), cropped_img), axis=1).astype(int)



    return cropped_img
def translate(x, y, dist):

    """x: 2d(m*n) numpy array. 1-dimension image data;

       y: 1d numpy array. The ground truth label;

       dist: float. Percentage of height/width to shift.

       return translated images.

       This function shift the image to 4 different directions.

       Crop a part of the image, shift it and fill the left part with 0."""

    # convert the 1d image data to a m*h*w*c array

    images = convert_2d(x)

    m, height, width, channel = images.shape

    

    # set 4 groups of anchors. The first 4 int in a certain group lay down the area we crop.

    # The last 4 sets the area to be moved to. E.g.,

    # new_img[new_top:new_bottom, new_left:new_right] = img[top:bottom, left:right]

    anchors = []

    anchors.append((0, height, int(dist * width), width, 0, height, 0, width - int(dist * width)))

    anchors.append((0, height, 0, width - int(dist * width), 0, height, int(dist * width), width))

    anchors.append((int(dist * height), height, 0, width, 0, height - int(dist * height), 0, width))

    anchors.append((0, height - int(dist * height), 0, width, int(dist * height), height, 0, width))

    

    # new_images: d*m*h*w*c array. The first dimension is the 4 directions.

    new_images = np.zeros((4, m, height, width, channel))

    for i in range(4):

        # shift the image

        top, bottom, left, right, new_top, new_bottom, new_left, new_right = anchors[i]

        new_images[i, :, new_top:new_bottom, new_left:new_right, :] = images[:, top:bottom, left:right, :]

    

    new_images = np.reshape(new_images, (4 * m, -1))

    y = np.tile(y, (4, 1)).reshape((-1, 1))

    new_images = np.concatenate((y, new_images), axis=1).astype(int)



    return new_images
def add_noise(x, y, noise_lvl):

    """x: 2d(m*n) numpy array. 1-dimension image data;

       y: 1d numpy array. The ground truth label;

       noise_lvl: float. Percentage of pixels to add noise in.

       return images with white noise.

       This function randomly picks some pixels and replace them with noise."""

    m, n = x.shape

    # calculate the # of pixels to add noise in

    noise_num = int(noise_lvl * n)



    for i in range(m):

        # generate n random numbers, sort it and choose the first noise_num indices

        # which equals to generate random numbers w/o replacement

        noise_idx = np.random.randint(0, n, n).argsort()[:noise_num]

        # replace the chosen pixels with noise from 0 to 255

        x[i, noise_idx] = np.random.randint(0, 255, noise_num)



    noisy_data = np.concatenate((y.reshape((-1, 1)), x), axis=1).astype("int")



    return noisy_data
def rotate_image(x, y, max_angle):

    """x: 2d(m*n) numpy array. 1-dimension image data;

       y: 1d numpy array. The ground truth label;

       max_angle: int. The maximum degree for rotation.

       return rotated images.

       This function rotates the image for some random degrees(0.5 to 1 * max_angle degree)."""

    images = convert_2d(x)

    m, height, width, channel = images.shape

    

    img_tensor = tf.placeholder(tf.float32, [m, height, width, channel])

    

    # half of the images are rotated clockwise. The other half counter-clockwise

    # positive angle: [max/2, max]

    # negative angle: [360-max/2, 360-max]

    rand_angle_pos = np.random.randint(max_angle / 2, max_angle, int(m / 2))

    rand_angle_neg = np.random.randint(-max_angle, -max_angle / 2, m - int(m / 2)) + 360

    rand_angle = np.transpose(np.hstack((rand_angle_pos, rand_angle_neg)))

    np.random.shuffle(rand_angle)

    # convert the degree to radian

    rand_angle = rand_angle / 180 * pi

    

    # rotate the images

    rotated_img_tensor = tf.contrib.image.rotate(img_tensor, rand_angle)



    with tf.Session() as sess:

        rotated_imgs = sess.run(rotated_img_tensor, feed_dict={img_tensor: images})

    

    rotated_imgs = np.reshape(rotated_imgs, (m, -1))

    rotated_imgs = np.concatenate((y.reshape((-1, 1)), rotated_imgs), axis=1)

    

    return rotated_imgs
start = time.clock()

print("Augment the data...")

cropped_imgs = crop_image(x_train, y_train, 0.9)

translated_imgs = translate(x_train, y_train, 0.1)

noisy_imgs = add_noise(x_train, y_train, 0.1)

rotated_imgs = rotate_image(x_train, y_train, 10)



data_train = np.vstack((data_train, cropped_imgs, translated_imgs, noisy_imgs, rotated_imgs))

np.random.shuffle(data_train)

print("Done!")

time_used = int(time.clock() - start)

print("Time used: {}s.".format(time_used))
x_train = data_train[:, 1:]

y_train = data_train[:, 0]

x_test = data_test.values

print("Augmented training data: {} rows, {} columns.".format(data_train.shape[0], data_train.shape[1]))
x_train = convert_2d(x_train)

x_test = convert_2d(x_test)
num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
x_train = x_train / 255

x_test = x_test / 255
# generate a random seed for train-test-split

seed = np.random.randint(1, 100)

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)
del data_train

del data_test

gc.collect()
# number of channels for each of the 4 convolutional layers. 

filters = (32, 32, 64, 64)

# I use a 5x5 kernel for every conv layer

kernel = (5, 5)

# the drop probability of the dropout layer

drop_prob = 0.2



model = keras.models.Sequential()



model.add(Conv2D(filters[0], kernel, padding="same", input_shape=(28, 28, 1),

                 kernel_initializer=keras.initializers.he_normal()))

model.add(BatchNormalization())

model.add(ReLU())

model.add(Conv2D(filters[0], kernel, padding="same",

                 kernel_initializer=keras.initializers.he_normal()))

model.add(BatchNormalization())

model.add(ReLU())

model.add(MaxPool2D())

model.add(Dropout(drop_prob))



model.add(Conv2D(filters[1], kernel, padding="same",

                 kernel_initializer=keras.initializers.he_normal()))

model.add(BatchNormalization())

model.add(ReLU())

model.add(MaxPool2D())

model.add(Dropout(drop_prob))



model.add(Conv2D(filters[2], kernel, padding="same",

                 kernel_initializer=keras.initializers.he_normal()))

model.add(BatchNormalization())

model.add(ReLU())

model.add(MaxPool2D())

model.add(Dropout(drop_prob))



model.add(Conv2D(filters[3], kernel, padding="same",

                 kernel_initializer=keras.initializers.he_normal()))

model.add(BatchNormalization())

model.add(ReLU())

model.add(MaxPool2D())

model.add(Dropout(drop_prob))



# several fully-connected layers after the conv layers

model.add(Flatten())

model.add(Dropout(drop_prob))

model.add(Dense(128, activation="relu"))

model.add(Dropout(drop_prob))

model.add(Dense(num_classes, activation="softmax"))

# use the Adam optimizer to accelerate convergence

model.compile(keras.optimizers.Adam(), "categorical_crossentropy", metrics=["accuracy"])
model.summary()
# number of epochs we run

iters = 100

# batch size. Number of images we train before we take one step in MBGD.

batch_size = 1024
# monitor: the quantity to be monitored. When it no longer improves significantly, we lower the learning rate

# factor: new learning rate = old learning rate * factor

# patience: number of epochs we wait before we decrease the learning rate

# verbose: whether or not the message are displayed

# min_lr: the minimum learning rate

lr_decay = ReduceLROnPlateau(monitor="val_acc", factor=0.5, patience=3, verbose=1, min_lr=1e-5)
# monitor: the quantity to be monitored. When it no longer improves significantly, stop training

# patience: number of epochs we wait before training is stopped

# verbose: whether or not to display the message

early_stopping = EarlyStopping(monitor="val_acc", patience=7, verbose=1)
print("Training model...")

fit_params = {

    "batch_size": batch_size,

    "epochs": iters,

    "verbose": 1,

    "callbacks": [lr_decay, early_stopping],

    "validation_data": (x_dev, y_dev)                   # data for monitoring the model accuracy

}

history = model.fit(x_train, y_train, **fit_params)

print("Done!")
train_acc = history.history["acc"]

val_acc = history.history["val_acc"]

train_loss = history.history["loss"]

val_loss = history.history["val_loss"]



plt.plot(train_acc)

plt.plot(val_acc)

plt.xlabel("epoch")

plt.ylabel("accuracy")

plt.legend(["train_acc", "val_acc"], loc="upper right")

plt.show()
plt.plot(train_loss)

plt.plot(val_loss)

plt.xlabel("epoch")

plt.ylabel("loss")

plt.legend(["train_loss", "val_loss"], loc="upper right")

plt.show()
loss, acc = model.evaluate(x_dev, y_dev)

print("Validation loss: {:.4f}".format(loss))

print("Validation accuracy: {:.4f}".format(acc))
num_samples = 10

dev_size = x_dev.shape[0]

sample_idx = np.random.randint(dev_size, size=num_samples)

x_samples = x_dev[sample_idx, :, :, :]

y_samples_pred = np.argmax(model.predict(x_samples), axis=1)



plt.figure(figsize=(20, 10))

for i in range(num_samples):

    

    plt.subplot(2, 5, i + 1)

    plt.imshow(x_samples[i, :, :, 0], cmap="gray")

    plt.title("Prediction: {}".format(y_samples_pred[i]))
y_val_true = np.argmax(y_dev, axis=1)

y_val_pred = np.argmax(model.predict(x_dev), axis=1)

conf_matrix = pd.DataFrame(confusion_matrix(y_val_true, y_val_pred), index=[i for i in range(10)], columns=[i for i in range(10)])

plt.figure(figsize = (10,7))

sbn.heatmap(conf_matrix, annot=True)
y_pred = model.predict(x_test, batch_size=batch_size)

y_pred = np.argmax(y_pred, axis=1).reshape((-1, 1))

idx = np.reshape(np.arange(1, len(y_pred) + 1), (len(y_pred), -1))

y_pred = np.hstack((idx, y_pred))

y_pred = pd.DataFrame(y_pred, columns=['ImageId', 'Label'])

y_pred.to_csv('y_pred.csv', index=False)