!pip3 install validator_collection
# https://www.tensorflow.org/tutorials/generative/style_transfer



import tensorflow as tf

import numpy as np

import PIL.Image

import matplotlib.pyplot as plt

# import IPython.display as display

# import matplotlib as mpl

# mpl.rcParams['figure.figsize'] = (12, 12)

# mpl.rcParams['axes.grid'] = False



# tf.enable_eager_execution()





from validator_collection import checkers, os



def load_img(path_to_img):

    if checkers.is_url(path_to_img):

        path_to_img = tf.keras.utils.get_file(os.path.basename(path_to_img), path_to_img)

    

    max_dim = 512

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

    if len(image.shape) > 3:

        image = tf.squeeze(image, axis=0)



    plt.imshow(image)

    if title:

        plt.title(title)

        

def tensor_to_image(tensor):

    tensor = tensor * 255

    tensor = np.array(tensor, dtype = np.uint8)

    if np.ndim(tensor) > 3:

        assert tensor.shape[0] == 1

        tensor = tensor[0]

    return PIL.Image.fromarray(tensor)



import tensorflow_hub as hub

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# content_image = load_img('https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/gettyimages-1038241336a-1587582575.jpg')

# content_image = load_img('https://www.rusnor.org/upload/main/693/my_photo.jpg')

content_image = load_img('http://m.seser.ru/img/arshinov.jpg')

# content_image = load_img('http://m.seser.ru/img/IMG_20170731_165949.jpg')



# style_image = load_img('https://i.pinimg.com/564x/05/83/6e/05836ef0e32e5c9cbe43e78c67f9b56a.jpg')

# style_image = load_img('https://i.pinimg.com/236x/32/87/82/328782ff45e59ae069ade7217da7bf59.jpg')

# style_image = load_img('https://i.pinimg.com/236x/22/9b/ba/229bbaaf985552a847e47af2cf275a0a.jpg')

# style_image = load_img('https://cs10.pikabu.ru/post_img/2019/02/11/4/15498644101100467833.jpg')

# style_image = load_img('https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')

# style_image = load_img('https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

# style_image = load_img('https://cs2.livemaster.ru/storage/ae/2c/65755e34850c7b388d02a1c8380q--kartiny-i-panno-interernaya-kartina-maslom-afrikanka.jpg')

# style_image = load_img('https://cs8.pikabu.ru/post_img/big/2017/12/04/10/1512409251120924912.jpg')

# style_image = load_img('https://artpostergallery.ru/userdata/image/thumbs/7d/0e/7d0e5815c760026836df83b7ddaa2f5a_2.jpg')

# style_image = load_img('https://98.img.avito.st/640x480/8174802698.jpg')

style_image = load_img('https://st3.depositphotos.com/5798150/12954/i/450/depositphotos_129540046-stock-photo-flowery-mood-original-oil-painting.jpg')



plt.subplot(1, 3, 1)

imshow(content_image, 'Content Image')

plt.subplot(1, 3, 2)

imshow(style_image, 'Style Image')



# import tensorflow_hub as hub

# hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]

tensor_to_image(stylized_image)

# plt.subplot(1,3,3)

# imshow(stylized_image, 'Output Image')
