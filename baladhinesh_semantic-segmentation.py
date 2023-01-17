# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
import tensorflow as tf



train_folder = "../input/cityscapes-image-pairs/cityscapes_data/train"

test_folder = "../input/cityscapes-image-pairs/cityscapes_data/val"





train_onlyfiles = [f for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]

test_onlyfiles = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]





print("Working with {0} train images".format(len(train_onlyfiles)))

print("Working with {0} test images".format(len(test_onlyfiles)))



from scipy import ndimage

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



train_files = []

test_files = []

train_img_arr = []

test_img_arr = []



i=0

for _file in train_onlyfiles:

    train_files.append(_file)

for _file in test_onlyfiles:

    test_files.append(_file)

    



for _file in train_files:

    img = load_img(train_folder + "/" + _file)  # this is a PIL image

    img.thumbnail((256, 512))

    # Convert to Numpy Array

    x = img_to_array(img) 

    train_img_arr.append(x)

for _file in test_files:

    img = load_img(test_folder + "/" + _file)  # this is a PIL image

    img.thumbnail((256, 512))

    # Convert to Numpy Array

    x = img_to_array(img) 

    test_img_arr.append(x)
train_arr = np.array(train_img_arr)

test_arr = np.array(test_img_arr)
x_train = train_arr[:,:,0:128,:]

y_train = train_arr[:,:,128:256,:]

x_test = test_arr[:,:,0:128,:]

y_test = test_arr[:,:,128:256,:]

x_train = x_train/255

y_train = y_train/255

x_test = x_test/255

y_test = y_test/255
import matplotlib.pyplot as plt



plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(x_train[i])

plt.show()
from keras.layers import Conv2D,MaxPooling2D,Dropout,Conv2DTranspose,concatenate
# # detect and init the TPU

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# tf.config.experimental_connect_to_cluster(tpu)

# tf.tpu.experimental.initialize_tpu_system(tpu)



# # instantiate a distribution strategy

# tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)



# # instantiating the model in the strategy scope creates the model on the TPU

# with tpu_strategy.scope():

inputs = tf.keras.layers.Input((128, 128, 3))

start_neurons = 16

dropout_rate = 0.4

conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)

conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)

pool1 = MaxPooling2D((2, 2))(conv1)

pool1 = Dropout(dropout_rate)(pool1)



conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)

conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)

pool2 = MaxPooling2D((2, 2))(conv2)

pool2 = Dropout(dropout_rate)(pool2)



conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)

conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)

pool3 = MaxPooling2D((2, 2))(conv3)

pool3 = Dropout(dropout_rate)(pool3)



conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)

conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)

pool4 = MaxPooling2D((2, 2))(conv4)

pool4 = Dropout(dropout_rate)(pool4)

    

conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)

conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(conv5)

pool5 = MaxPooling2D((2, 2))(conv5)

pool5 = Dropout(dropout_rate)(pool5)



# Middle

convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(pool5)

convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(convm)

    

deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)

uconv5 = concatenate([deconv5, conv5])

uconv5 = Dropout(dropout_rate)(uconv5)

uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)

uconv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv5)



deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv5)

uconv4 = concatenate([deconv4, conv4])

uconv4 = Dropout(dropout_rate)(uconv4)

uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)



deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)

uconv3 = concatenate([deconv3, conv3])

uconv3 = Dropout(dropout_rate)(uconv3)

uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)



deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)

uconv2 = concatenate([deconv2, conv2])

uconv2 = Dropout(dropout_rate)(uconv2)

uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    

deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)

uconv1 = concatenate([deconv1, conv1])

uconv1 = Dropout(dropout_rate)(uconv1)

uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    

outputs = Conv2D(3, (1,1), padding="same", activation="sigmoid")(uconv1)

    

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

    







history = model.fit(x_train, y_train, epochs=40, 

                    validation_data=(x_test, y_test))

probability_model = tf.keras.Sequential([model, 

                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)
predictions[0].shape
tf.keras.preprocessing.image.array_to_img(

    predictions[1]

)
tf.keras.preprocessing.image.array_to_img(

    y_test[1]

)