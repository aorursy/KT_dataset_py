# plot feature map of first conv layer for given image

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array

from keras.models import Model

from matplotlib import pyplot

from numpy import expand_dims

import numpy as np

import keras.backend as K

import tensorflow as tf



K.clear_session()

# load the model

model = VGG16()

model.summary()



# load the image with the required shape

img = load_img('../input/treebird/bird.jpg', target_size=(224, 224))

# convert the image to an array

img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'

img = expand_dims(img, axis=0)

# prepare the image (e.g. scale pixel values for the vgg)

img = preprocess_input(img)



preds_raw = model.predict(img)

print("dtype of preds_raw is", type(preds_raw))

print("since preds_raw is numpy we use .shape to determine its shape")

print("the shape of the preds_raw is", preds_raw.shape)

print(model.output.eval(session=K.get_session(), feed_dict={'input_1:0': img}))#that's how to determine the value of tensor   

print("model.output datatype is", type(model.output))

print("eventhough model.output is a tensor we use .shape to determine its shape")

print("shape of model.output is", model.output.shape)

sess = K.get_session()

shapes = tf.shape(model.output)

print("dynamic shape of model.output is",sess.run(shapes, feed_dict={'input_1:0': img}))

#comparing the shape of the two meethods of producing 2nd layer output

import keras.backend as K



outputs=model.layers[1].output

print("data type of outputs is",type(outputs))

print("eventhough datatype of outputs is a tensor we use .shape to determine its shape")

print("the shape of the outputs is", outputs.shape)

sess = K.get_session()

shapes = tf.shape(outputs)

print("dynamic shape of model.output is",sess.run(shapes, feed_dict={'input_1:0': img}))

#print(outputs.eval(session=K.get_session(), feed_dict={'input_1:0': img})) #that's how to determine the value of tensor

square = 8

ix = 1

for _ in range(square):

    for _ in range(square):

        # specify subplot and turn of axis

        ax = pyplot.subplot(square, square, ix)

        ax.set_xticks([])

        ax.set_yticks([])

        # plot filter channel in grayscale

        np_array = outputs[0, :, :, ix-1].eval(session=K.get_session(),feed_dict={'input_1:0': img})

        pyplot.imshow(np_array, cmap='gray')

        ix += 1

# show the figure

pyplot.show()

#1- compare the layer output (outputs) and feature_map

#2- plot the layer output (outputs)

model1 = Model(inputs=model.inputs, outputs=model.layers[1].output)

feature_maps = model1.predict(img)

print("data type of feature_maps is",type(feature_maps))

#print(feature_maps)

square = 8

ix = 1

for _ in range(square):

    for _ in range(square):

        # specify subplot and turn of axis

        ax = pyplot.subplot(square, square, ix)

        ax.set_xticks([])

        ax.set_yticks([])

        # plot filter channel in grayscale

        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')

        ix += 1

# show the figure

pyplot.show()
