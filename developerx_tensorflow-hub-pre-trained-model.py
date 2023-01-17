import tensorflow as tf

from IPython.display import Image

from IPython.core.display import HTML



import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12,12)

mpl.rcParams['axes.grid'] = False



import numpy as np

import PIL.Image
def tensor_to_image(tensor):



    tensor = tensor*255

    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor)>3:

        assert tensor.shape[0] == 1

        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)
content_path_orig = '../input/gan-getting-started/photo_jpg/00068bc07f.jpg'

style_path_orig = '../input/gan-getting-started/monet_jpg/000c1e3bff.jpg'
def load_img(path_to_img):

    max_dim = 256

    img = tf.io.read_file(path_to_img)

    img = tf.image.decode_image(img, channels=3)

    img = tf.image.convert_image_dtype(img, tf.float32)



    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    long_dim = max(shape)

    scale = max_dim / long_dim



    new_shape = tf.cast(shape * scale, tf.int32)



    img = tf.image.resize(img, new_shape)

    img = img[tf.newaxis, :]

    return img
def imshow(image, title=None):

    # convert tensor elements to 0 - 255 values

    image = image * 255

    

    # Remove batch dimension of tensor

    if len(image.shape) > 3:

        out = np.squeeze(image, axis=0)

     

    out = out.astype('uint8')

    plt.imshow(out)

    if title:

        plt.title(title)

    plt.imshow(out)
plt.figure(figsize=(10,10))



content_image_orig = load_img(content_path_orig)

style_image_orig = load_img(style_path_orig)



plt.subplot(1, 2, 1)

imshow(content_image_orig, 'Content Image')



plt.subplot(1, 2, 2)

imshow(style_image_orig, 'Style Image')
import tensorflow_hub as hub

# This is version is updated from the original notebook (/2 vs. /1)

# Results are greatly improved

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image_orig = hub_module(tf.constant(content_image_orig), tf.constant(style_image_orig))[0]

plt.figure(figsize=(15,15))



plt.subplot(1, 3, 1)

imshow(content_image_orig, 'Content Image')



plt.subplot(1, 3, 2)

imshow(stylized_image_orig, 'Result')



plt.subplot(1, 3, 3)

imshow(style_image_orig, 'Style Image')
style_path_flip = '../input/gan-getting-started/photo_jpg/00068bc07f.jpg'

content_path_flip = '../input/gan-getting-started/monet_jpg/000c1e3bff.jpg'



content_image_flip = load_img(content_path_flip)

style_image_flip = load_img(style_path_flip)



stylized_image_flip = hub_module(tf.constant(content_image_flip), tf.constant(style_image_flip))[0]

plt.figure(figsize=(15,15))



plt.subplot(1, 3, 1)

imshow(content_image_flip, 'Content Image')



plt.subplot(1, 3, 2)

imshow(stylized_image_flip, 'Result')



plt.subplot(1, 3, 3)

imshow(style_image_flip, 'Style Image')
from os import listdir

from os.path import isfile, join





photo_path = '../input/gan-getting-started/photo_jpg/'

photo_files = [f for f in listdir(photo_path) if isfile(join(photo_path, f))]



monet_path = '../input/gan-getting-started/monet_jpg/'

monet_files = [f for f in listdir(monet_path) if isfile(join(monet_path, f))]

monet_count = len(monet_files)
from random import randrange





# Get random photo

content_path1 = photo_path + photo_files[randrange(len(photo_files))]

style_path1 = monet_path + monet_files[randrange(len(monet_files))]



content_image1 = load_img(content_path1)

style_image1 = load_img(style_path1)



content_image2 = style_image1

style_image2 = content_image1



stylized_image1 = hub_module(tf.constant(content_image1), tf.constant(style_image1))[0]

stylized_image2 = hub_module(tf.constant(content_image2), tf.constant(style_image2))[0]



plt.figure(figsize=(20,20))

plt.subplots_adjust(bottom=0.3, top=0.7, hspace=0.25, right=0.8, wspace=0.3)



plt.subplot(2, 3, 1)

imshow(content_image1, 'Content Image')



plt.subplot(2, 3, 2)

imshow(stylized_image1, 'Result')



plt.subplot(2, 3, 3)

imshow(style_image1, 'Style Image')



plt.subplot(2, 3, 4)

imshow(content_image2, 'Content Image')



plt.subplot(2, 3, 5)

imshow(stylized_image2, 'Result')



plt.subplot(2, 3, 6)

imshow(style_image2, 'Style Image')