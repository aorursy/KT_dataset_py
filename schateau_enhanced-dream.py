import numpy as np

import time

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import scipy

import warnings

warnings.filterwarnings('ignore', '.*output shape of zoom.*')



from keras.preprocessing.image import load_img, save_img, img_to_array

from keras.applications import inception_v3, VGG19

from keras import backend as K

from keras.layers import Input



from scipy.optimize import fmin_l_bfgs_b



from PIL import Image

import PIL
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
iterations = 15



# dimensions of the generated picture.

img_width = 450

img_height = 900



#Create a dict assigning weights for the layers we want to excite:

layer_contributions = {

    'mixed2': 0.2,

    'mixed3': 2.0,

    'mixed4': 2.5,

    'mixed5': 3.0  

}



coeff_continuity_loss = 0.1

coeff_l2_loss = 0.1

coeff_jitter = 0.0



base_image_path = '/kaggle/input/testpainting/birth.jpg'

#base_image_path = '/kaggle/input/images-to-test/Shark.jpg'

#base_image_path = '/kaggle/input/images-to-test/Night_sky.jpg'
img = load_img(base_image_path)

display(img.resize((img_height,img_width), PIL.Image.LANCZOS))
#Compute the img_size variable:

if K.image_data_format() == 'channels_first':

    img_size = (3, img_width, img_height)

else:

    img_size = (img_width, img_height, 3)
def resize_img(img, size):

    img = np.copy(img)

    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)

    return scipy.ndimage.zoom(img, factors, order=1)



#Utility functions for opening, resizing and formating image into tensors for the model

def preprocess_image_InceptionV3(image_path):

    img = load_img(image_path, target_size=(img_width, img_height))

    img = img_to_array(img)

    img = np.expand_dims(img, axis=0)

    #preprocess image the same way as in InceptionV3

    img = inception_v3.preprocess_input(img)

    return img

# util functions to convert a tensor into a valid image

def deprocess_image_InceptionV3(x):

    if K.image_data_format() == 'channels_first':

        x = x.reshape((3, img_width, img_height))

        x = x.transpose((1, 2, 0))

    else:

        x = x.reshape((img_width, img_height, 3))     

    x /= 2.

    x += 0.5

    x *= 255.

    x = np.clip(x, 0, 255).astype('uint8')

    return x
#The dream will be stored in this tensor:

dream = Input(batch_shape=(1,) + img_size)



#disable the training of the model

K.set_learning_phase(0)



#load the model with pretrained image weights and only the convolutional base

path_to_inceptionv3 = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = inception_v3.InceptionV3(input_tensor=dream,

                                 weights=path_to_inceptionv3,

                                 include_top=False)

print('Model loaded.')



layer_dict = dict([(layer.name, layer) for layer in model.layers])
# define the loss

loss = K.variable(0.)

for layer_name in layer_contributions:

    # add the L2 norm of the features of a layer to the loss

    assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'

    coeff = layer_contributions[layer_name]

    #retrieves the layer's output and shape:

    activation = layer_dict[layer_name].output

    shape = layer_dict[layer_name].output_shape

    #Define a "scaling" variable so that the contribution from larger layers does not outweigh smaller layers.

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    # we avoid border artifacts by only involving non-border pixels in the loss

    # note: we use substraction here, because the "content loss" is defined as K.sum(K.square(dream-base)). We are computing "-base" here

    if K.image_data_format() == 'channels_first':

        loss = loss - coeff * K.sum(K.square(activation[:, :, 2: shape[2] - 2, 2: shape[3] - 2])) / scaling 

    else:

        loss = loss - coeff * K.sum(K.square(activation[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / scaling
#tweaks for better results:

# continuity loss util function

def continuity_loss(x):

    assert K.ndim(x) == 4

    if K.image_data_format() == 'channels_first':

        a = K.square(x[:, :, :img_width - 1, :img_height - 1] -

                     x[:, :, 1:, :img_height - 1])

        b = K.square(x[:, :, :img_width - 1, :img_height - 1] -

                     x[:, :, :img_width - 1, 1:])

    else:

        a = K.square(x[:, :img_width - 1, :img_height-1, :] -

                     x[:, 1:,             :img_height-1, :]) # (pixel i- pixel i_1)^2

        b = K.square(x[:, :img_width - 1, :img_height-1, :] -

                     x[:, :img_width - 1, 1:,            :]) # (pixel j- pixel j_1)^2

    return K.sum(K.pow(a + b, 1.25))



# add a "total variation loss" or "continuity loss" which operatess on the pixels of generated dream: it encourages spatial continuity, thus avoiding overly pixelated results.

loss = loss + coeff_continuity_loss * continuity_loss(dream) / np.prod(img_size)

# add image L2 norm to loss (prevent pixels from taking very high values). Somahow similar as content loss

loss = loss + coeff_l2_loss * K.sum(K.square(dream)) / np.prod(img_size)
#This class wraps fetch_loss_and_grads in a way that lets you retrieve the losses and the gradients via two separate methods calls, which is required by the SciPy optimizer.

class Evaluator(object):

    def __init__(self):

        self.loss_value = None

        self.grad_values = None



    def loss(self, x):

        assert self.loss_value is None

        #loss_value, grad_values = eval_loss_and_grads(x)

        x = x.reshape((1,) + img_size)

        outs = fetch_loss_and_grads([x])

        loss_value = outs[0]

        if len(outs[1:]) == 1:

            grad_values = outs[1].flatten().astype('float64')

        else:

            grad_values = np.array(outs[1:]).flatten().astype('float64')

        self.loss_value = loss_value

        self.grad_values = grad_values

        return self.loss_value



    def grads(self, x):

        assert self.loss_value is not None

        grad_values = np.copy(self.grad_values)

        self.loss_value = None

        self.grad_values = None

        return grad_values



evaluator = Evaluator()
# compute the gradients of the dream wrt the loss

grads = K.gradients(loss, dream)[0]



outputs = [loss]

if type(grads) in {list, tuple}:

    outputs += grads

else:

    outputs.append(grads)

#Set up Keras function to retrieve the loss and gradient values given an input image:

fetch_loss_and_grads = K.function([dream], outputs)
x = preprocess_image_InceptionV3(base_image_path)



for i in range(iterations):

    start_time = time.time()

    # add a random jitter to the initial image. This will be reverted at decoding time

    random_jitter = (coeff_jitter * 2) * (np.random.random(img_size) - 0.5)

    x += random_jitter

    

    # run L-BFGS for 7 steps

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss,

                                     x.flatten(), #because L-BFGS algo only process flat vectors

                                     fprime=evaluator.grads,

                                     maxfun=5)

    # decode the dream and save it

    x = x.reshape(img_size)

    x -= random_jitter

    img = deprocess_image_InceptionV3(np.copy(x))        

    end_time = time.time()

    print('Iteration %d completed in %ds' % (i, end_time - start_time), 'Current loss value:', min_val)

    if (i)%5==0:

        img_pil = Image.fromarray(img, 'RGB')

        display(img_pil.resize((img_height,img_width), PIL.Image.LANCZOS))

        

img_pil = Image.fromarray(img, 'RGB')

display(img_pil.resize((img_height,img_width), PIL.Image.LANCZOS))