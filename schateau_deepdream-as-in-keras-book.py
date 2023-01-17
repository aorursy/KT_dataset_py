import numpy as np

from datetime import datetime as dt

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



import scipy

import warnings

warnings.filterwarnings('ignore', '.*output shape of zoom.*')



from keras.preprocessing.image import load_img, save_img, img_to_array

from keras.applications import inception_v3

from keras import backend as K



from PIL import Image

import PIL



#from IPython.display import Image



#import random

#import tensorflow as tf
#disable the training of the model

K.set_learning_phase(0)



#load the model with pretrained image weights and only the convolutional base

path_to_inceptionv3 = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

model = inception_v3.InceptionV3(weights=path_to_inceptionv3,

                                 include_top=False)
def resize_img(img, size):

    img = np.copy(img)

    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)

    return scipy.ndimage.zoom(img, factors, order=1)



#Utility function for opening, resizing and formating image into tensors for the model

def preprocess_image(image_path, target_size=None):

    #load image

    img = load_img(image_path, target_size=target_size)

    #convert to array

    img = img_to_array(img)

    #add an additional dim for 'N'

    img = np.expand_dims(img, axis=0)

    #preprocess image the same way as in InceptionV3

    img = inception_v3.preprocess_input(img)

    return img



def deprocess_image(x):

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
#We'll do a weighted sum of the L2 norm of these layers. Create dict first:

layer_contributions = {

    'mixed2': 0.2,

    'mixed3': 3.0,

    #'mixed4': 2.0,

    'mixed5': 1.5    

}
#Create dict that maps layer_names -> layer instances

layer_dict = dict([(layer.name, layer) for layer in model.layers])



#The loss is defined by adding the contribution of each layer to the "loss" variable (scalar):

loss = K.variable(0.)

for layer_name in layer_contributions:

    coeff = layer_contributions[layer_name]

    #retrieves the layer's output

    activation = layer_dict[layer_name].output

    #Define a "scaling" variable as the product of activation's shape

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))

    #The loss is defined for NHWC outputs (i.e. channel_last)

    #it is normalizaed at each layer so the contribution from larger layers does not outweigh smaller layers.

    loss = loss + coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling
#The dream will be stored in this tensor:

dream = model.input

#Computes the gradients of the dream with regard to the loss:

grads = K.gradients(loss, dream)[0]

#Normalize the gradients (helps the gradient-ascent process to go smoothly):

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7) #add 1e-7 to avoid dividing by zero

#Set up Keras function to retrieve the loss and gradient values given an input image:

outputs = [loss, grads]

fetch_loss_and_grads = K.function([dream], outputs)



def eval_loss_and_grads(x):

    outs = fetch_loss_and_grads([x])

    loss_value = outs[0]

    grad_values = outs[1]

    return loss_value, grad_values



def gradient_ascent(x, iterations, step, max_loss=None):

    for i in range(iterations):

        loss_value, grad_values = eval_loss_and_grads(x)

        if max_loss is not None and loss_value > max_loss:

            break

        print('...Loss value at', i, ':', loss_value)

        #This is gradient ascent:

        x += step*grad_values

        

        ##visualisation:

        #if i%10==0:

        #    print('dream at iterations', i)

        #    pil_img = deprocess_image(np.copy(x))

        #    pil_img = Image.fromarray(pil_img, 'RGB')

        #    display(pil_img.resize((500, 300), PIL.Image.LANCZOS))

    return x
#!Not used right now, work in progress...



#Applying random shifts to the image before each tiled computation prevents tile seams from appearing.



#def random_roll(img, maxroll):

#    random_seed = 213

#    # Randomly shift the image to avoid tiled boundaries:

#    shift = K.random_uniform_variable(shape=[2], low=-maxroll, high=maxroll, dtype='int32', seed=random_seed)

#    shift_down, shift_right = shift[0],shift[1]

#    img_rolled = tf.roll(tf.roll(img, shift_right, axis=2), shift_down, axis=1)

#    return shift_down, shift_right, img_rolled



#shift_down, shift_right, img_rolled = random_roll(img, maxroll=512)
#Not working: work in progress...



#def gradient_ascent_tiled(img, iterations, step, max_loss=None, tile_size=128):

#    for i in range(iterations):

#        shift_down, shift_right, img_rolled = random_roll(img, tile_size)

#        # Initialize the image gradients to zero.

#        gradients = tf.zeros_like(img_rolled)

#        for x in range(0, img_rolled.shape[1], tile_size):

#            for y in range(0, img_rolled.shape[2], tile_size):

#                # Extract a tile out of the image.

#                img_tile = img_rolled[x:x+tile_size, y:y+tile_size]

#                loss_value_img_rolled, grad_values_img_rolled = eval_loss_and_grads(img_tile)

#                #if max_loss is not None and loss_value_img_rolled > max_loss:

#                #    break

#                print('...Loss value at', i, ':', loss_value_img_rolled)

#                # Update the image gradients for this tile.

#                gradients = gradients + grad_values_img_rolled

#        # Undo the random shift applied to the image and its gradients.

#        gradients = tf.roll(tf.roll(gradients, -shift_right, axis=2), -shift_down, axis=1)

#

#        #This is gradient ascent:

#        img += step*gradients

#    return x
base_image_path = '/kaggle/input/images-to-test/Night_sky.jpg'
plt.figure(figsize=[12,12])

plt.title("Original image",fontsize=20)

img = load_img(base_image_path)

plt.imshow(img)
# Downsizing the image makes it easier to work with.

img = preprocess_image(base_image_path, target_size=[550, 980])
start = dt.now()



#Hyperparameters to play with for cool effects:

step = 0.01 #gradient ascent step size

num_octave = 3 #number of scales

octave_scale = 1.4 #size ratio between scales

iterations = 30 #number of ascent steps to run at each scale/octave



#defin a maximal loss to avoid ugly artifact:

max_loss = 10.



#exclude the first dimension for obtaining the image shape

original_shape = img.shape[1:3]

successive_shapes = [original_shape]

for i in range(1, num_octave):

    shape = tuple([int(dim/(octave_scale**i)) for dim in original_shape])

    successive_shapes.append(shape)

    

#reverse the list of shapes, so they are in increasing order:

successive_shapes = successive_shapes[::-1]

#save the img into original_img

original_img = np.copy(img)



#resizes the Numpy array of the image to the smallest octave:

shrunk_original_img = resize_img(img, successive_shapes[0])



for i,shape in enumerate(successive_shapes):

    print('Octave #', i, ' - Processing image shape', shape)

    #scales up the dream image

    img = resize_img(img,shape)

    #run gradient ascent on the dream:

    img = gradient_ascent(img, 

                          iterations=iterations, 

                          step=step, 

                          max_loss=max_loss)

    

    #upscale the shrunked version of the original image: it will be pixellated

    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)

    #Computes the high-quality version of the original image for the same shape

    same_size_original = resize_img(original_img, shape)

    #The difference between the two is the detail that was lost when scaling up:

    lost_detail = same_size_original - upscaled_shrunk_original_img

    

    #add the lost detail to the new image

    img += lost_detail

    #update the shrunk_original_image to last shape:

    shrunk_original_img = resize_img(original_img, shape)



stop = dt.now()

print("\n", (stop - start).seconds, "seconds")

print("Done.")



print('Final dream')

pil_img = deprocess_image(np.copy(img))

plt.figure(figsize = (12,12))

plt.imshow(pil_img)