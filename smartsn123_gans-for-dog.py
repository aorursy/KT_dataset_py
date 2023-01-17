import os

root_dir = ""

project_dir =  ""
#!mkdir gdrive/'My Drive'/ML/APTOS2019/

#!ls {project_dir}
#!cp {project_dir}/dogs_gan.ipynb gdrive/'My Drive'/ML/APTOS2019/
data_dir = "../input/"
!ls {data_dir}

#!unzip {data_dir}/all-dogs.zip -d {data_dir}

#!unzip {data_dir}/Annotation.zip -d {data_dir}

#!ls
import sys

#sys.path.append("gdrive/My Drive/ML/")

#import download_utils
import tensorflow as tf

import keras

from keras import backend as K

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import cv2  # for image processing

import scipy.io

import os

#import keras_utils

#from keras_utils import reset_tf_session 

print(tf.__version__)

print(keras.__version__)
plt.rcParams.update({'axes.titlesize': 'small'})
def reset_tf_session():

    curr_session = tf.get_default_session()

    # close current session

    if curr_session is not None:

        curr_session.close()

    # reset graph

    K.clear_session()

    # create new session

    config = tf.ConfigProto()

    config.gpu_options.allow_growth = True

    s = tf.InteractiveSession(config=config)

    K.set_session(s)

    return s
IMG_SIZE = 128
#!ls ../input/all-dogs/all-dogs
dogs_imgages = "../input/all-dogs/all-dogs/"

annotations = "../input/annotation/"
#!ls {data_dir}/all-dogs
import glob
images = [f for f in glob.glob(dogs_imgages + "*.jpg")]

annotations = [f for f in glob.glob(annotations + "*/n*")]
print (images[:5])

print (annotations[:2])
from PIL import Image

from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET

import random

images_to_display = random.choices(images, k=64)



fig = plt.figure(figsize=(25, 16))

for ii, img in enumerate(images_to_display):

    ax = fig.add_subplot(8, 8, ii + 1, xticks=[], yticks=[])

    img = cv2.imread(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #img = Image.open(img_byte)

    plt.imshow(img)
img = cv2.imread(images[100])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



print(img.shape)

#cv2_imshow(img)

fig = plt.figure(figsize=(4,8))

plt.imshow(img)

plt.show()
def get_bbox(dog_id):

  breed_folder = [x.rsplit('/', 1)[0] for x in annotations if dog_id in x.split('/')[-1] ]

  #print (breed_folder)

  if len(breed_folder) != 1:

    return None

  breed_folder = breed_folder[0]

  file_name = "{}/{}".format(breed_folder, dog_id)

  #print (file_name)

  root = ET.parse(file_name).getroot()

  objects = root.findall('object')

  for obj in objects:

      bndbox = obj.find('bndbox')

      xmin = int(bndbox.find('xmin').text)

      ymin = int(bndbox.find('ymin').text)

      xmax = int(bndbox.find('xmax').text)

      ymax = int(bndbox.find('ymax').text)

  bbox = (xmin, ymin, xmax, ymax)

  #print("Bounding Box: ", bbox)

  return bbox

  

  
dog_id = 'n02109961_16718'
box = get_bbox(dog_id)
def get_annotated_img(img_file):

  dog_id = img_file.split('/')[-1].split('.')[0]

  #raw_bytes = read_raw_from_zip(zip_dogs,"all-dogs/{}.jpg".format(dog_id))

  #img = decode_image_from_raw_bytes(raw_bytes)

  img = cv2.imread(img_file)

  bbox = get_bbox(dog_id)

  if bbox:

    xmin, ymin, xmax, ymax = bbox

    img = img[ymin:ymax,xmin:xmax]

  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

  img = img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  return img

  
img = get_annotated_img(images[4])

img.shape
fig = plt.figure()

plt.imshow(img)

plt.show()
import tensorflow as tf

#from keras_utils import reset_tf_session

s = reset_tf_session()



import keras

from keras.models import Sequential

from keras import layers as L
import random

from scipy import ndarray

import skimage as sk

from skimage import transform

from skimage import util

from copy import deepcopy

def random_rotation(image_array: ndarray):

    # pick a random degree of rotation between 25% on the left and 25% on the right

    random_degree = random.uniform(-25, 25)

    return sk.transform.rotate(image_array, random_degree)



def random_noise(image_array: ndarray):

    # add random noise to the image

    return sk.util.random_noise(image_array)



def horizontal_flip(image_array: ndarray):

    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !

    return image_array[:, ::-1]
def generate_training_images(images, batch_size=100):

  cur_batch = []

  for image in images:

    img = get_annotated_img(image)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE) ) 

    cur_batch.append(img)

    if len(cur_batch) == batch_size:

      yield cur_batch

      cur_batch = []

    cur_batch.append(random_rotation(deepcopy(img) ))

    if len(cur_batch) == batch_size:

      yield cur_batch

      cur_batch = []

    cur_batch.append(random_noise(deepcopy(img) ))

    if len(cur_batch) == batch_size:

      yield cur_batch

      cur_batch = []

    cur_batch.append(horizontal_flip(deepcopy(img) ))

    if len(cur_batch) == batch_size:

      yield cur_batch

      cur_batch = []

  return cur_batch
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
CODE_SIZE =256

dim = 8

depth = 64*4

dropout = 0.4



generator = Sequential()



#layer 1  : Dense Layer with BatchNormalization and relu activation

generator.add(L.Dense(dim * dim * depth, 

                      input_shape=(CODE_SIZE,)))

generator.add(L.LeakyReLU(alpha=0.2))

generator.add(L.BatchNormalization(momentum=0.8))





#layer2 :  Reshape and Dropout

generator.add(L.Reshape((dim, dim, depth)))

#generator.add(L.Dropout(dropout))





#layer 3 : Transpose conv layer

generator.add(L.Deconv2D(int(depth/2), 

                         kernel_size=(5,5)))

generator.add(L.BatchNormalization(momentum=0.8))

generator.add(L.LeakyReLU(alpha=0.2))









#layer 4 : upsample conv layer

generator.add(L.UpSampling2D())

generator.add(L.Deconv2D(int(depth/4), 

                         kernel_size=(5,5)))

generator.add(L.Dropout(dropout))

generator.add(L.BatchNormalization(momentum=0.8))

generator.add(L.LeakyReLU(alpha=0.2))





generator.add(L.Deconv2D(int(depth/4), 

                         kernel_size=(5,5)))

generator.add(L.Dropout(dropout))

generator.add(L.BatchNormalization(momentum=0.8))

generator.add(L.LeakyReLU(alpha=0.2))





#layer 5 : upsample conv layer

generator.add(L.UpSampling2D(size=(2,2)))

generator.add(L.Deconv2D(int(depth/4), 

                         kernel_size=(5,5)))

generator.add(L.Dropout(dropout))

generator.add(L.BatchNormalization(momentum=0.8))

generator.add(L.LeakyReLU(alpha=0.2))



#layer 5 : Image generation Layer

generator.add(L.Conv2D(3, kernel_size=5, activation=None))

generator.add(L.UpSampling2D())
#generator = load_model('./G')

generator.summary()
discriminator = Sequential()

depth = 64

dropout = 0.4



discriminator.add(L.InputLayer(IMG_SHAPE))



discriminator.add(L.Conv2D(depth, (3, 3),padding='same') )

discriminator.add(L.LeakyReLU(0.1))



discriminator.add(L.Conv2D(depth*2, (3, 3)))

discriminator.add(L.LeakyReLU(0.1))

discriminator.add(L.Dropout(dropout))



discriminator.add(L.MaxPool2D())



discriminator.add(L.Conv2D(depth*3, (3, 3)))

discriminator.add(L.LeakyReLU(0.1))



discriminator.add(L.MaxPool2D())



discriminator.add(L.Conv2D(depth, (3, 3)))

discriminator.add(L.LeakyReLU(0.1))

discriminator.add(L.Dropout(dropout))



discriminator.add(L.MaxPool2D())

discriminator.add(L.Flatten())

discriminator.add(L.Dense(512,activation='tanh'))

discriminator.add(L.Dense(CODE_SIZE,activation='tanh'))

discriminator.add(L.Dense(2, activation=tf.nn.log_softmax))
#discriminator = load_model('./D')

discriminator.summary()
noise = tf.placeholder('float32',[None,CODE_SIZE])

real_data = tf.placeholder('float32',[None,]+list(IMG_SHAPE))



logp_real = discriminator(real_data)



generated_data = generator(noise)



logp_gen = discriminator(generated_data)
########################

#discriminator training#

########################



d_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])



#regularize

d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)



#optimize

disc_optimizer =  tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss, var_list=discriminator.trainable_weights)
########################

###generator training###

########################



g_loss = -tf.log(1-logp_gen)



gen_optimizer = tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)



    
def sample_noise_batch(bsize):

    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')



def sample_data_batch(bsize):

    idxs = np.random.choice(np.arange(len(images)), size = int(bsize/2) )

    cur_batch = []

    for idx in idxs:

        img = get_annotated_img(images[idx]) 

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE) )

        #print (img.shape)

        cur_batch.append( img)

        

    idxs = np.random.choice(np.arange(len(images)), size= bsize-int(bsize/2) )

    for idx in idxs:

        img = get_annotated_img(images[idx]) 

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE) )

        if idxs[0]%3 == 0:

          cur_batch.append( random_rotation(img))

        elif idxs[0]%3 == 1:

          cur_batch.append( random_noise(img))

        else:

          cur_batch.append( horizontal_flip(img))

        #print (img.shape)

    return np.array(cur_batch)



def sample_images(nrow,ncol, sharp=False):

    images = generator.predict(sample_noise_batch(bsize=nrow*ncol))

    for i in range(nrow*ncol):

        plt.subplot(nrow,ncol,i+1)

        if sharp:

            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray", interpolation="none")

        else:

            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray")

            

    plt.show()



def sample_probas(bsize):

    plt.title('Generated vs real data')

    plt.hist(np.exp(discriminator.predict(sample_data_batch(bsize)))[:,1],

             label='D(x)', alpha=0.5,range=[0,1])

    plt.hist(np.exp(discriminator.predict(generator.predict(sample_noise_batch(bsize))))[:,1],

             label='D(G(z))',alpha=0.5,range=[0,1])

    plt.legend(loc='best')

    plt.show()
#import tqdm_utils
#!ls {data_dir}

s.run(tf.global_variables_initializer())
from time import time

init_time = time()
from tqdm import tqdm_notebook as tqdm
from IPython import display



for epoch in tqdm(range(50000)):

    

    feed_dict = {

        real_data:sample_data_batch(100),

        noise:sample_noise_batch(100)

    }

    

    for i in range(5):

        s.run(disc_optimizer, feed_dict)

    

    s.run(gen_optimizer,feed_dict)

    fl = 0

    if epoch %100==0 and epoch > 0:

        display.clear_output(wait=True)

        sample_images(2,3,True)

        sample_probas(1000)

        if epoch%500 ==0 and epoch > 0:

            generator.save('G_{}'.format(epoch))

            discriminator.save('D_{}'.format(epoch))

            if int((time() - init_time)/3600) > 6:

                fl = 1

                break

    if fl:

        break

        

        