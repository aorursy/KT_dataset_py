import functools

import os



from matplotlib import gridspec

import matplotlib.pylab as plt

import numpy as np

import tensorflow as tf

import tensorflow_hub as hub
print("TF Version: ", tf.__version__)

print("TF-Hub version: ", hub.__version__)

print("Eager mode enabled: ", tf.executing_eagerly())

print("GPU available: ", tf.test.is_gpu_available())
def crop_center(image):

  """Returns a cropped square image."""

  shape = image.shape

  new_shape = min(shape[1], shape[2])

  offset_y = max(shape[1] - shape[2], 0) // 2

  offset_x = max(shape[2] - shape[1], 0) // 2

  image = tf.image.crop_to_bounding_box(

      image, offset_y, offset_x, new_shape, new_shape)

  return image



@functools.lru_cache(maxsize=None)

def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):

  """Loads and preprocesses images."""

  # Cache image file locally.

  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)

  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].

  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]

  if img.max() > 1.0:

    img = img / 255.

  if len(img.shape) == 3:

    img = tf.stack([img, img, img], axis=-1)

  img = crop_center(img)

  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)

  return img



def show_n(images, titles=('',)):

  n = len(images)

  image_sizes = [image.shape[1] for image in images]

  w = (image_sizes[0] * 6) // 320

  plt.figure(figsize=(w  * n, w))

  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)

  for i in range(n):

    plt.subplot(gs[i])

    plt.imshow(images[i][0], aspect='equal')

    plt.axis('off')

    plt.title(titles[i] if len(titles) > i else '')

  plt.show()
output_image_size = 384 

content_img_size = (output_image_size, output_image_size)

style_img_size = (256, 256)
content_image_url = 'https://d16yj43vx3i1f6.cloudfront.net/uploads/2019/10/GettyImages-803849852.jpg'

style_image_url = 'https://vertexpages.com/wp-content/uploads/2019/10/farm.jpg'
content_image = load_image(content_image_url, content_img_size)

style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

show_n([content_image, style_image], ['Content image', 'Style image'])
import time

start_time = time.time()
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

hub_module = hub.load(hub_handle)
outputs = hub_module(content_image, style_image)

stylized_image = outputs[0]
# Stylize content image with given style image.

# This is pretty fast within a few milliseconds on a GPU.



outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

stylized_image = outputs[0]
# Visualize input images and the generated stylized image.



show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
end_time = time.time()

print('Time Taken = ', end_time-start_time)
content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Taj_Mahal_%28Edited%29.jpeg/1920px-Taj_Mahal_%28Edited%29.jpeg'

style_image_url = 'https://joeburciaga.files.wordpress.com/2013/02/tsunami-2698.jpg'
content_image = load_image(content_image_url, content_img_size)

style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

show_n([content_image, style_image], ['Content image', 'Style image'])
start_time = time.time()
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

hub_module = hub.load(hub_handle)
outputs = hub_module(content_image, style_image)

stylized_image = outputs[0]
# Stylize content image with given style image.

# This is pretty fast within a few milliseconds on a GPU.



outputs = hub_module(tf.constant(content_image), tf.constant(style_image))

stylized_image = outputs[0]
# Visualize input images and the generated stylized image.



show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
end_time = time.time()

print('Time Taken = ', end_time-start_time)