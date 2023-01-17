# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf 





# setting the dimensions of the input images

batch_size = 128

no_classes = 10

epochs = 2

image_height, image_width = 28, 28



# loading data from disk to memory using keras

(x_train, y_train), (x_test, y_test) = "../input/mnist-data"  # tf.keras.datasets.mnist.load_data()



# reshaping the vector into image format && defining input dim for convolution

x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)



x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)



input_shape_tuple = (image_height, image_width, 1)



# converting datatype to float32

x_train = x_train.astype('float32')

x_test = x_test.astype('float32')



# data normalization by subtracting mean of data

x_train /= 255

x_test /= 255



# converting categorical labels to one-shot encoding

y_train = tf.keras.utils.to_categorical(y_train, no_classes)

y_test = tf.keras.utils.to_categorical(y_test, no_classes)



# starting the session and initializing the variables

session = tf.Session()

session.run(tf.global_variables_initializer())



# the modal

def simple_cnn(input_shape_tuple):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(

        filters=64,

        kernel_size=(3, 3),

        activation='relu',

        input_shape=input_shape_tuple

    ))

    model.add(tf.keras.layers.Conv2D(

        filters=128,

        kernel_size=(3, 3),

        activation='relu'

    ))

    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=1024, activation='relu'))

    model.add(tf.keras.layers.Dropout(rate=0.3))

    model.add(tf.keras.layers.Dense(units=no_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model



simple_cnn_model = simple_cnn(input_shape_tuple)



# loading training data with the training parameters & fit the model

simple_cnn_model.fit(x_train, y_train, batch_size, epochs, (x_test, y_test))

train_loss, train_accuracy = simple_cnn_model.evaluate(x_train, y_train, verbose=0)

print('Train data loss: ', train_loss)

print('Train data accuracy: ', train_accuracy)



# data evaluation

test_loss, test_accuracy = simple_cnn_model.evaluate(

    x_test, y_test, verbose=0

)

print('Test data loss:', test_loss)

print('Test data accuracy:', test_accuracy)





work_dir = "../input"  # working directory



image_names = sorted(os.listdir(os.path.join(work_dir, "train")))





def copy_files(prefix_str, range_start, range_end, target_dir):

    image_paths = [

        os.path.join(work_dir, "data", target_dir, prefix_str + "." + str(i) + ".jpg")

        for i in range(range_start, range_end)

    ]

    dest_dir = os.path.join(work_dir, "data", target_dir, prefix_str)

    os.makedirs(dest_dir)

    for image_path in image_paths:

        shutil.copy(image_path, dest_dir)





copy_files("dog", 0, 1000, "train")

copy_files("cat", 0, 1000, "train")

copy_files("dog", 1000, 1400, "test")

copy_files("cat", 1000, 1400, "test")



# benchmarking with simple cnn



image_height, image_width = 150, 150

train_dir = os.path.join(work_dir, "train")

test_dir = os.path.join(work_dir, "test")

no_classes = 2

no_validation = 800

epochs = 2

batch_size = 200

no_train = 2000

no_test = 800

input_shape = (image_height, image_width, 3)

epoch_steps = no_train // batch_size

test_steps = no_test // batch_size





"""

generator_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

generator_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)



"""

generator_train = tf.keras.preprocessing.image.IMageDataGenerator(

    rescale=1. / 255,

    horizontal_flip=True,

    zoom_range=0.3,

    shear_range=0.3

)





train_images = generator_train.flow_from_directory(

    test_dir, batch_size=batch_size, target_size=(image_width, image_height)

)





# fitting data to the model

simple_cnn_model.fit_generator(

    train_images,

    steps_per_epoch=epoch_steps,

    epochs=epochs,

    validation_data=test_images,

    validation_steps=test_steps,

)





# transfer learning

## training on bottleneck features



generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

model = tf.keras.applications.VGG16(include_top=False)



train_images = generator.flow_from_directory(

    train_dir,

    batch_size=batch_size,

    target_size=(image_width, image_height),

    class_mode=None,

    shuffle=False,

)

train_bottleneck_features = model.predict_generator(train_images, epoch_steps)



test_images = generator.flow_from_directory(

    test_dir,

    batch_size=batch_size,

    target_size=(image_width, image_height),

    classMode=None,

    shuffle=False,

)



test_bottleneck_features = model.predict_generator(test_images, test_steps)





## sequential model for prediction



train_labels = np.array([0] * int(no_train / 2) + [1] * int(no_train / 2))

test_labels = np.array([0] * int(no_test / 2) + [1] * int(no_test / 2))



# bottleneck feature implementation



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=train_bottleneck_features.shape[1:1]))

model.add(tf.keras.layers.Dense(1024, activation="relu"))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation="softmax"))

model.compile(

    loss=tf.keras.losses.categorical_crossentropy,

    optimizer=tf.keras.optimizers.Adam(),

    metrics=["accuracy"],

)



# training the bottleneck features

model.fit(

    train_bottleneck_features,

    train_labels,

    batch_size=batch_size,

    epochs=epochs,

    validation_data=(test_bottleneck_features, test_labels),

)





#### fine tuning several layers



top_model_weights_path = "fc_model.h5"



# loading the Visual Geometry Group (VGG) model



model = tf.keras.applications.VGG16(include_top=False)



# small two-layer feedforward network on top of the VGG



model_fine_tune = tf.keras.models.Sequential()

model_fine_tune.add(tf.keras.layers.Flatten(input_shape=model.output_shape))

model_fine_tune.add(tf.keras.layers.Dense(256, activation="relu"))

model_fine_tune.add(tf.keras.layers.Dropout(0.5))

model_fine_tune.add(tf.keras.layers.Dense(no_classes, activation="softmax"))





# loading the top model with pre-trained weights



model_fine_tune.load_weights(top_model_weights_path)

model.add(model_fine_tune)





# adding the top model to the convolution base



for vgg_layer in model.layers[:25]:

    vgg_layer.trainable = False





# compiling the model with gradient descent optimizer

# @ slow learning rate magnitude order 4



model.compile(

    loss="binary_crossentropy",

    optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),

    metrics=["accuracy"],

)


