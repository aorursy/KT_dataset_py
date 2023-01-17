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
import numpy as np

import scipy

import PIL.Image



import tensorflow as tf

from tensorflow import keras

from keras.preprocessing.image import load_img, save_img, img_to_array

from keras.applications import inception_v3

from keras.applications import vgg16



from keras import backend as K



# To fix FailedPreconditionError:

sess = tf.InteractiveSession()

with tf.Session() as sess:

     sess.run(tf.global_variables_initializer()) 
'''

# Disable all training specific operations

K.set_learning_phase(0)





# The model will be loaded with pre-trained inceptionv3 weights.

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

dream = model.input

print('Model loaded.')



# You can tweak these setting to obtain new visual effects.

settings = {

    'features': {

        'mixed2': 0.4,

        'mixed3': 1.5,

        'mixed4': 2.,

        'mixed5': 3.1,

    },

}

'''









K.set_learning_phase(0)





# The model will be loaded with pre-trained inceptionv3 weights.

model = vgg16.VGG16(weights='imagenet', include_top=False)

dream = model.input

print('Model loaded.')

settings = {

    'features': {

        'block4_conv2': 0.4,

        'block4_conv3': 1.5,

        'block5_conv1': 2.,

        'block5_conv2': 3.1,

    },

}





# Set a function to load, resize and convert the image. 

def preprocess_image(image_path):

    # Util function to open, resize and format pictures

    # into appropriate tensors.

    img = load_img(image_path)

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = inception_v3.preprocess_input(img)

    return img





# And a function to do the opposite: convert a tensor into an image. 

def deprocess_image(x):

    # Util function to convert a tensor into a valid image.

    if K.image_data_format() == 'channels_first':

        x = x.reshape((3, x.shape[2], x.shape[3]))

        x = x.transpose((1, 2, 0))

    else:

        x = x.reshape((x.shape[1], x.shape[2], 3))

    x /= 2.

    x += 0.5

    x *= 255.

    x = np.clip(x, 0, 255).astype('uint8')

    return x





# Set a dictionary that maps the layer name to the layer instance. 

# get the symbolic outputs of each "key" layer (we gave them unique names).

layer_dict = dict([(layer.name, layer) for layer in model.layers])





# Define the loss. The way this works is first the scalar variable *loss* is set. 

# Then the loss will be defined by adding layer contributions to this variable. 

loss = K.variable(0.)



for layer_name in settings['features']:

    # Add the L2 norm of the features of a layer to the loss.

    assert (layer_name in layer_dict.keys(),

            'Layer ' + layer_name + ' not found in model.')

    coeff = settings['features'][layer_name]

    x = layer_dict[layer_name].output

    # We avoid border artifacts by only involving non-border pixels in the loss.

    scaling = K.prod(K.cast(K.shape(x), 'float32'))

    if K.image_data_format() == 'channels_first':

        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling

    else:

        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# Compute the gradients of the dream wrt the loss.

grads = K.gradients(loss, dream)[0]

# Normalize gradients.

grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())



# Set up function to retrieve the value of the loss and gradients given an input image.

outputs = [loss, grads]

fetch_loss_and_grads = K.function([dream], outputs)



def eval_loss_and_grads(x):

    outs = fetch_loss_and_grads([x])

    loss_value = outs[0]

    grad_values = outs[1]

    return loss_value, grad_values



# Helper funtion to resize

def resize_img(img, size):

    img = np.copy(img)

    if K.image_data_format() == 'channels_first':

        factors = (1, 1,

                   float(size[0]) / img.shape[2],

                   float(size[1]) / img.shape[3])

    else:

        factors = (1,

                   float(size[0]) / img.shape[1],

                   float(size[1]) / img.shape[2],

                   1)

    return scipy.ndimage.zoom(img, factors, order=1)





# Define the gradient ascent function over a number of iterations. 

def gradient_ascent(x, iterations, step, max_loss=None):

    for i in range(iterations):

        loss_value, grad_values = eval_loss_and_grads(x)

        if max_loss is not None and loss_value > max_loss:

            break

        print('..Loss value at', i, ':', loss_value)

        x += step * grad_values

    return x





# Set hyperparameters. The ocatave_scale is the ratio between each successive scale (remember the upscaling mentioned before?). 

# Playing with these hyperparameters will also allow you to achieve new effects

step = 0.009  # Gradient ascent step size

num_octave = 7  # Number of scales at which to run gradient ascent

octave_scale = 1.4  # Size ratio between scales

iterations = 40  # Number of ascent steps per scale

max_loss = 60.

print(os.listdir("../input"))
base_image_path = "../input/photo-1535498730771-e735b998cd64.jpg"

img = PIL.Image.open(base_image_path)

img
img = preprocess_image(base_image_path)

if K.image_data_format() == 'channels_first':

    original_shape = img.shape[2:]

else:

    original_shape = img.shape[1:3]

successive_shapes = [original_shape]

for i in range(1, num_octave):

    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])

    successive_shapes.append(shape)

successive_shapes = successive_shapes[::-1]

original_img = np.copy(img)

shrunk_original_img = resize_img(img, successive_shapes[0])



for shape in successive_shapes:

    print('Processing image shape', shape)

    img = resize_img(img, shape)

    img = gradient_ascent(img,

                          iterations=iterations,

                          step=step,

                          max_loss=max_loss)

    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)

    same_size_original = resize_img(original_img, shape)

    lost_detail = same_size_original - upscaled_shrunk_original_img



    img += lost_detail

    shrunk_original_img = resize_img(original_img, shape)



save_img('dream.jpg',deprocess_image(np.copy(img)))

dreamout = PIL.Image.open('dream.jpg')

dreamout
type(dreamout)
dreamout.save("hello.jpg")