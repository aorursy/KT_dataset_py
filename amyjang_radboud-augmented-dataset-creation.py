! pip install tensorflow==2.2.0 -q
import os

import PIL

import math

import warnings

import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split



SEED = 1337

print('Tensorflow version : {}'.format(tf.__version__))



try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

except ValueError:

    strategy = tf.distribute.get_strategy() # for CPU and single GPU

    

print('Number of replicas:', strategy.num_replicas_in_sync)
MAIN_DIR = '../input/pandatilesagg'

TRAIN_IMG_DIR = os.path.join(MAIN_DIR, 'all_images')

train_csv = pd.read_csv(os.path.join(MAIN_DIR, 'train.csv'))
radboud_csv = train_csv[train_csv['data_provider'] == 'radboud']

karolinska_csv = train_csv[train_csv['data_provider'] != 'radboud']

img_ids = train_csv['image_id']
r_train, r_test = train_test_split(

    radboud_csv,

    test_size=0.2, random_state=SEED

)



k_train, k_test = train_test_split(

    karolinska_csv,

    test_size=0.2, random_state=SEED

)
AUTOTUNE = tf.data.experimental.AUTOTUNE

IMG_DIM = (1536, 128)

CLASSES_NUM = 6

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

EPOCHS = 100

N=12
def decode_img(img):

  # convert the compressed string to a 3D uint8 tensor

  img = tf.image.decode_jpeg(img, channels=3)

  # Use `convert_image_dtype` to convert to floats in the [0,1] range.

  img = tf.image.convert_image_dtype(img, tf.float32)

  # resize the image to the desired size.

  return tf.image.resize(img, IMG_DIM)
def get_item(file_path):    

    image = tf.io.read_file(file_path)

    image = decode_img(image)

    label = tf.strings.split(file_path, '_')

    label = tf.strings.to_number(label[-2])

    label = tf.cast(label, tf.int32)

    

    return image, tf.one_hot(label, CLASSES_NUM)
r_train['isup_grade'] = r_train['isup_grade'].apply(str)

r_train['file'] = TRAIN_IMG_DIR +'/_' + r_train['isup_grade'] + '_' + r_train['image_id'] + '.jpg'



r_test['isup_grade'] = r_test['isup_grade'].apply(str)

r_test['file'] = TRAIN_IMG_DIR +'/_' + r_test['isup_grade'] + '_' + r_test['image_id'] + '.jpg'



k_train['isup_grade'] = k_train['isup_grade'].apply(str)

k_train['file'] = TRAIN_IMG_DIR +'/_' + k_train['isup_grade'] + '_' + k_train['image_id'] + '.jpg'



k_test['isup_grade'] = k_test['isup_grade'].apply(str)

k_test['file'] = TRAIN_IMG_DIR +'/_' + k_test['isup_grade'] + '_' + k_test['image_id'] + '.jpg'
def get_dataset(df):

    ds = tf.data.Dataset.from_tensor_slices(df['file'].values)

    ds = ds.map(get_item, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds
r_train_ds = get_dataset(r_train)

k_train_ds = get_dataset(k_train)
def show_batch(image_batch, label_batch):

    plt.figure(figsize=(10,10))

    for n in range(10):

        ax = plt.subplot(1,10,n+1)

        plt.imshow(image_batch[n])

        plt.title(np.amax(label_batch[n].numpy()))

        plt.axis("off")
image_batch, _ = next(iter(r_train_ds))

r_image = image_batch[4].numpy()

plt.figure(figsize=(10,10))

ax = plt.subplot(1,2,1)

plt.title("radboud")

plt.imshow(r_image)

plt.axis("off")



image_batch, _ = next(iter(k_train_ds))

k_image = image_batch[4].numpy()

ax = plt.subplot(1,2,2)

plt.imshow(k_image)

plt.title("karolinska")

plt.axis("off")
content_image = tf.expand_dims(r_image, axis = 0)

style_image = tf.expand_dims(k_image, axis = 0)
content_layers = ['block5_conv2'] 



style_layers = ['block1_conv1',

                'block2_conv1',

                'block3_conv1', 

                'block4_conv1', 

                'block5_conv1']



num_content_layers = len(content_layers)

num_style_layers = len(style_layers)
def vgg_layers(layer_names):

  """ Creates a vgg model that returns a list of intermediate output values."""

  # Load our model. Load pretrained VGG, trained on imagenet data

  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

  vgg.trainable = False

  

  outputs = [vgg.get_layer(name).output for name in layer_names]



  model = tf.keras.Model([vgg.input], outputs)

  return model
style_extractor = vgg_layers(style_layers)

style_outputs = style_extractor(style_image*255)
def gram_matrix(input_tensor):

  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)

  input_shape = tf.shape(input_tensor)

  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)

  return result/(num_locations)
class StyleContentModel(tf.keras.models.Model):

  def __init__(self, style_layers, content_layers):

    super(StyleContentModel, self).__init__()

    self.vgg =  vgg_layers(style_layers + content_layers)

    self.style_layers = style_layers

    self.content_layers = content_layers

    self.num_style_layers = len(style_layers)

    self.vgg.trainable = False



  def call(self, inputs):

    "Expects float input in [0,1]"

    inputs = inputs*255.0

    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

    outputs = self.vgg(preprocessed_input)

    style_outputs, content_outputs = (outputs[:self.num_style_layers], 

                                      outputs[self.num_style_layers:])



    style_outputs = [gram_matrix(style_output)

                     for style_output in style_outputs]



    content_dict = {content_name:value 

                    for content_name, value 

                    in zip(self.content_layers, content_outputs)}



    style_dict = {style_name:value

                  for style_name, value

                  in zip(self.style_layers, style_outputs)}

    

    return {'content':content_dict, 'style':style_dict}
extractor = StyleContentModel(style_layers, content_layers)



results = extractor(tf.constant(content_image))
style_targets = extractor(style_image)['style']

content_targets = extractor(content_image)['content']
def clip_0_1(image):

  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight=1e-1

content_weight=1e4
def style_content_loss(outputs):

    style_outputs = outputs['style']

    content_outputs = outputs['content']

    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 

                           for name in style_outputs.keys()])

    style_loss *= style_weight / num_style_layers



    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 

                             for name in content_outputs.keys()])

    content_loss *= content_weight / num_content_layers

    loss = style_loss + content_loss

    return loss
#@tf.function()

def train_step(image):

  with tf.GradientTape() as tape:

    outputs = extractor(image)

    loss = style_content_loss(outputs)



  grad = tape.gradient(loss, image)

  opt.apply_gradients([(grad, image)])

  image.assign(clip_0_1(image))
steps_per_epoch = 3



def style_transfer(image):

    image = tf.expand_dims(image, axis = 0)

    image = tf.Variable(lambda : image)

    step = 0

    for m in range(steps_per_epoch):

        step += 1

        train_step(image)

        

    print(".", end='')

    return image
plt.figure(figsize=(10,10))

ax = plt.subplot(1,2,1)

plt.title("radboud")

plt.imshow(r_image)

plt.axis("off")



r_aug_image = style_transfer(r_image)

ax = plt.subplot(1,2,2)

plt.imshow(r_aug_image[0])

plt.title("radboud augmented")

plt.axis("off")
! mkdir images
for file in r_train['file'].values:

    img = np.array(PIL.Image.open(file)) /255.0

    img = style_transfer(img)[0] * 255.0

    img = img.numpy()

    im = img.astype('uint8')

    im = PIL.Image.fromarray(im)

    im.save("images/" + file.split('/')[-1])
import shutil

shutil.make_archive("images", 'zip', "/kaggle/working/images")