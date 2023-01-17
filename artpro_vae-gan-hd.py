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
from IPython.display import clear_output

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from keras.layers import Dropout, BatchNormalization, Reshape, Flatten, RepeatVector

from keras.layers import Lambda, Dense, Input, Conv2D, MaxPool2D, UpSampling2D, concatenate

from keras.layers.advanced_activations import LeakyReLU

from keras.layers import Activation

from keras.models import Model, load_model





# Регистрация сессии в keras

from keras import backend as K

import tensorflow as tf

sess = tf.Session()

K.set_session(sess)
import os

import numpy as np

import pandas as pd

import random

from tqdm import tqdm

import shutil



def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])



ComputeLB = False

DogsOnly = True



import numpy as np, pandas as pd, os

import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



ROOT = '../input/generative-dog-images/'

if not ComputeLB: ROOT = '../input/'

IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))

imagesIn2 = np.zeros((25000,32,32,3))

# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# https://www.kaggle.com/paulorzp/show-annotations-and-breeds

if DogsOnly:

    for breed in breeds:

        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

            except: continue           

            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                w = np.min((xmax - xmin, ymax - ymin))

                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

                #img3 = img2.convert('LA')

                img3 = img2

                #img2 = img2.convert('LA')

                #img2 = img2.resize((64,64), Image.ANTIALIAS)

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                img3 = img3.resize((32,32), Image.ANTIALIAS)

                #img3 = rgb2gray(np.asarray(img2)) 

                #imagesIn[idxIn,:,:] = np.asarray(img2)[:,:,0]

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                #imagesIn2[idxIn,:,:] = np.asarray(img3)[:,:,0]

                imagesIn2[idxIn,:,:,:] = np.asarray(img3)

                #if idxIn%1000==0: print(idxIn)

                namesIn.append(breed)

                idxIn += 1

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

#     imagesIn = imagesIn[idx,:,:,:]

#     imagesIn2 = imagesIn2[idx,:,:]

    imagesIn = imagesIn[idx,:,:]

    imagesIn2 = imagesIn2[idx,:,:]

    namesIn = np.array(namesIn)[idx]
#plt.imshow(imagesIn[101,:,:], cmap='Greys')

plt.imshow(Image.fromarray( (imagesIn[101]).astype('uint8').reshape((64,64,3))))
plt.imshow(Image.fromarray( (imagesIn2[101]).astype('uint8').reshape((32,32,3))))
def gen_batch(x, y):

    n_batches = x.shape[0] // batch_size

    while(True):

        idxs = np.random.permutation(y.shape[0])

        x = x[idxs]

        y = y[idxs]

        for i in range(n_batches):

            yield x[batch_size*i: batch_size*(i+1)], y[batch_size*i: batch_size*(i+1)]
from keras.utils import to_categorical

import sklearn.preprocessing

L_enc = sklearn.preprocessing.LabelEncoder()

y_train_cat = L_enc.fit_transform(namesIn)
y_train_cat = to_categorical(y_train_cat).astype(np.float32)
x_train = imagesIn2.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 32,32, 1))

x_train = np.reshape(x_train, (len(x_train), 32,32, 3))


batch_size = 64

Ndim = 32



batch_shape = (batch_size, Ndim, Ndim, 3)

latent_dim = 256

num_classes = 120

dropout_rate = 0.3

gamma = 1 





train_batches_it = gen_batch(x_train, y_train_cat)
x_ = tf.placeholder(tf.float32, shape=(None, Ndim, Ndim, 3),  name='image')

y_ = tf.placeholder(tf.float32, shape=(None, 120),         name='labels')

z_ = tf.placeholder(tf.float32, shape=(None, latent_dim), name='z')



img = Input(tensor=x_)

lbl = Input(tensor=y_)

z   = Input(tensor=z_)


def add_units_to_conv2d(conv2, units):

    dim1 = int(conv2.shape[1])

    dim2 = int(conv2.shape[2])

    #dim3 = int(conv2.shape[3]) #!

    dimc = int(units.shape[1])

    repeat_n = dim1*dim2

    units_repeat = RepeatVector(repeat_n)(lbl)

    units_repeat = Reshape((dim1, dim2, dimc))(units_repeat)

    return concatenate([conv2, units_repeat])





def apply_bn_relu_and_dropout(x, bn=False, relu=True, dropout=True):

    if bn:

        x = BatchNormalization(momentum=0.99, scale=False)(x)

    if relu:

        x = LeakyReLU()(x)

    if dropout:

        x = Dropout(dropout_rate)(x)

    return x





with tf.variable_scope('encoder'):

    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same')(img)

    #x = Conv2D(32, kernel_size=(3), strides=(2, 2), padding='same')(img)

    x = apply_bn_relu_and_dropout(x)

    x = MaxPool2D((2, 2), padding='same')(x)



    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)

    #x = Conv2D(64, kernel_size=(3), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    

    #x = MaxPool2D((2, 2), padding='same')(x)



    #x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)

    #x = Conv2D(128, kernel_size=(3), padding='same')(x)

    #x = apply_bn_relu_and_dropout(x)



    x = Flatten()(x)

    x = concatenate([x, lbl])

    

    h = Dense(512)(x) 

    h = apply_bn_relu_and_dropout(h)



    z_mean    = Dense(latent_dim)(h)

    z_log_var = Dense(latent_dim)(h)



    def sampling(args):

        z_mean, z_log_var = args

        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)

        return z_mean + K.exp(K.clip(z_log_var/2, -2, 2)) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

encoder = Model([img, lbl], [z_mean, z_log_var, l], name='Encoder')





with tf.variable_scope('decoder'):

    x = concatenate([z, lbl])



    x = Dense(64*64*2)(x)

    x = apply_bn_relu_and_dropout(x)

    #x = Reshape((4, 4,16))(x)

    x = Reshape((16, 16, 32))(x)

    x = UpSampling2D(size=(2,2))(x)



    x = Conv2D(128, kernel_size=(5,5), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)



    x = Conv2D(64, kernel_size=(3,3), padding='same')(x)

    #x = UpSampling2D(size=(2, 2))(x)

    x = apply_bn_relu_and_dropout(x)



#     x = Conv2D(32, kernel_size=(3,3), padding='same')(x)

#     x = UpSampling2D(size=(2, 2))(x)

#     x = apply_bn_relu_and_dropout(x)

    

    x = Conv2D(32, kernel_size=(3,3), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    decoded = Conv2D(3, kernel_size=(5,5), activation='sigmoid', padding='same')(x)



decoder = Model([z, lbl], decoded, name='Decoder')





with tf.variable_scope('discrim'):

    x = Conv2D(128, kernel_size=(7, 7), strides=(2, 2), padding='same')(img)

    x = MaxPool2D((2, 2), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    x = add_units_to_conv2d(x, lbl)



    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)

    x = MaxPool2D((2, 2), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)



    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)

    x = MaxPool2D((2, 2), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)

    x = MaxPool2D((2, 2), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    

    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)

    x = MaxPool2D((2, 2), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)

    # l-слой на котором будем сравнивать активации

    l = Conv2D(32, kernel_size=(3, 3), padding='same')(x)

    x = apply_bn_relu_and_dropout(x)



    h = Flatten()(x)

    d = Dense(1, activation='sigmoid')(h)



discrim = Model([img, lbl], [d, l], name='Discriminator')
z_mean, z_log_var, encoded_img = encoder([img, lbl])



decoded_img = decoder([encoded_img, lbl])

decoded_img.shape
z_mean, z_log_var, encoded_img = encoder([img, lbl])



decoded_img = decoder([encoded_img, lbl])

decoded_z   = decoder([z,           lbl])



discr_img,     discr_l_img     = discrim([img,         lbl])

discr_dec_img, discr_l_dec_img = discrim([decoded_img, lbl])

discr_dec_z,   discr_l_dec_z   = discrim([decoded_z,   lbl])



cvae_model = Model([img, lbl], decoder([encoded_img, lbl]), name='cvae')

cvae =  cvae_model([img, lbl])


L_prior = -0.5*tf.reduce_sum(1. + tf.clip_by_value(z_log_var, -2, 2) - tf.square(z_mean) \

                             - tf.exp(tf.clip_by_value(z_log_var, -2, 2)))/Ndim/Ndim



log_dis_img     = tf.log(discr_img + 1e-10)

log_dis_dec_z   = tf.log(1. - discr_dec_z + 1e-10)

log_dis_dec_img = tf.log(1. - discr_dec_img + 1e-10)



L_GAN = -1/4*tf.reduce_sum(log_dis_img + 2*log_dis_dec_z + log_dis_dec_img)/Ndim/Ndim



# L_dis_llike = tf.reduce_sum(tf.square(discr_l_img - discr_l_dec_img))/28/28

L_dis_llike = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.sigmoid(discr_l_img),

                                                                    logits=discr_l_dec_img))/Ndim/Ndim







L_enc = L_dis_llike + L_prior 

L_dec = gamma * L_dis_llike - L_GAN

L_dis = L_GAN







optimizer_enc = tf.train.RMSPropOptimizer(0.001)

optimizer_dec = tf.train.RMSPropOptimizer(0.0006)#0.0003

optimizer_dis = tf.train.RMSPropOptimizer(0.001)



encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")

decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")

discrim_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discrim")



step_enc = optimizer_enc.minimize(L_enc, var_list=encoder_vars)

step_dec = optimizer_dec.minimize(L_dec, var_list=decoder_vars)

step_dis = optimizer_dis.minimize(L_dis, var_list=discrim_vars)





def step(image, label, zp):

    l_prior, dec_image, l_dis_llike, l_gan, _, _ = sess.run([L_prior, decoded_z, L_dis_llike, L_GAN, step_enc, step_dec],

                                                            feed_dict={z:zp, img:image, lbl:label, K.learning_phase():1})

    return l_prior, dec_image, l_dis_llike, l_gan



def step_d(image, label, zp):

    l_gan, _ = sess.run([L_GAN, step_dis], feed_dict={z:zp, img:image, lbl:label, K.learning_phase():1})

    return l_gan
sess.run(tf.global_variables_initializer())

save_periods = list(range(100)) + list(range(100, 1000, 10))

nb_step = 3 



batches_per_period = 300

for i in range(30000):

    print('.', end='')





    for j in range(nb_step):

        b0, b1 = next(train_batches_it)

        zp = np.random.randn(batch_size, latent_dim)

        l_g = step_d(b0, b1, zp)

        if l_g < 1.0:

            break

        



    for j in range(nb_step):

        l_p, zx, l_d, l_g = step(b0, b1, zp)

        if l_g > 0.4:

            break

        b0, b1 = next(train_batches_it)

        zp = np.random.randn(batch_size, latent_dim)





    if not i % batches_per_period:

        period = i // batches_per_period



        print(i, l_p, l_d, l_g)
import os

import sys

import random

import warnings



import numpy as np

import pandas as pd

import cv2



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

import skimage

from PIL import Image

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.util import crop, pad

from skimage.morphology import label

from skimage.color import rgb2gray, gray2rgb, rgb2lab, lab2rgb

from sklearn.model_selection import train_test_split



from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from keras.models import Model, load_model,Sequential

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape, Embedding

from keras.layers.core import Dropout, Lambda

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.layers.pooling import MaxPooling2D

from keras.layers.merge import concatenate

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K



import tensorflow as tf
IMG_WIDTH = 32

IMG_HEIGHT = 32

IMG_CHANNELS = 3

INPUT_SHAPE=(IMG_HEIGHT, IMG_WIDTH, 3)



def Colorize():

    in_label = Input(shape=(1,))

    embed_input = Embedding(120, 1000)(in_label)

    

    

    #Encoder

    encoder_input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3,))

    encoder_output = Conv2D(32, (3,3), activation='relu', padding='same',strides=1)(encoder_input)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(32, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(32, (3,3), activation='relu', padding='same',strides=1)(encoder_output)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same',strides=1)(encoder_output)

    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)

    encoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(encoder_output)

    encoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(encoder_output)

    

    #Fusion

    #fusion_output = RepeatVector(2 * 2)(embed_input) 

    fusion_output = Dense(1024, activation='relu')(embed_input) 

    fusion_output = Reshape(([4, 4, 64]))(fusion_output)

    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 

    fusion_output = Conv2D(64, (1, 1), activation='relu', padding='same')(fusion_output)

    

    #Decoder

    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(fusion_output)

    decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    decoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(64, (2,2), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    

    decoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)

    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoder_output)

    decoder_output = UpSampling2D((2, 2))(decoder_output)

    return Model(inputs=[encoder_input, in_label], outputs=decoder_output)



model = Colorize()

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
from keras.utils import to_categorical

import sklearn.preprocessing

L_enc = sklearn.preprocessing.LabelEncoder()

labels = L_enc.fit_transform(namesIn)

labels = np.array(labels).reshape((len(labels),1))
shift = 6

y_train = imagesIn / 256.

datagen = ImageDataGenerator(

        shear_range=0.2,

        zoom_range=0.2,

        rotation_range=20,

        #width_shift_range=shift, 

        #height_shift_range=shift,

        #zca_whitening=True,

    

        horizontal_flip=True)



def image_a_b_gen(dataset=[labels, y_train], batch_size = 20):

    datagen.fit(y_train)

    for y_batch, labels_batch  in datagen.flow(y_train, labels, batch_size=batch_size):

        



        images = [Image.fromarray( (256.*Y).astype('uint8').reshape((64,64,3))) for Y in y_batch]

        img = [(I.resize((32,32), Image.ANTIALIAS)) for I in images]

        X_batch = np.array([np.asarray(I)[:,:] for I in img])/256.0

        

        yield [X_batch, labels_batch], y_batch
X_train = imagesIn2 / 256.

learning_rate_reduction = ReduceLROnPlateau(monitor='loss', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5,

                                            min_lr=0.00001)

filepath = "Art_Colorization_Model.h5"

checkpoint = ModelCheckpoint(filepath,

                             save_best_only=True,

                             monitor='loss',

                             mode='min')



model_callbacks = [learning_rate_reduction,checkpoint]



BATCH_SIZE = 20

model.fit_generator(image_a_b_gen([labels,y_train],BATCH_SIZE),

            epochs=30,

            verbose=1,

            steps_per_epoch=y_train.shape[0]/BATCH_SIZE,

             callbacks=model_callbacks

                   )
# X_batch = X_train

# y_train = imagesIn / 256.

# y_train = np.reshape(y_train, (len(y_train), 128,128, 1))

# model.fit([X_batch,labels],y_train,

#             epochs=1,

#             verbose=1)
def get_rgb(img):

    color_me_embed = np.array(np.random.randint(120))

    img = np.asarray(img)

    img = img.reshape((32,32,3))/256.0

    color_me_embed = color_me_embed.reshape(1,1)

    #print(color_me.shape)

    output = model.predict([[img], color_me_embed.reshape(1,1)])

    

    output = output*256.

    return Image.fromarray( (output).astype('uint8').reshape((64,64,3)))



from scipy.stats import norm

def image_gen():

    label = np.random.randint(120)

    

    input_lbl = np.zeros((1, 120))

    input_lbl[0, label] = 1

    xi = norm.ppf(np.linspace(0.05, 0.95, 1))

    yi = norm.ppf(np.linspace(0.05, 0.95, 1))



    z_sample = np.zeros((1, latent_dim))

    z_sample[:, :2] = np.array([[xi[0], yi[0]]])



    x_decoded = sess.run(decoded_z, feed_dict={z:z_sample, lbl:input_lbl, K.learning_phase():0})

    img = x_decoded[0].squeeze()

    return Image.fromarray((256*img).astype('uint8'))
img = image_gen()

img2 = get_rgb(img)

plt.imshow(img)

plt.imshow(img2)
import zipfile

my_zipfile = zipfile.PyZipFile('images.zip', mode='w')



for k in range(10000):



    img = image_gen()

    img = get_rgb(img)

    f = str(k)+'.png'

    img.save(f,'PNG')

    my_zipfile.write(f)

    os.remove(f)

    #if k % 1000==0: print(k)

my_zipfile.close()