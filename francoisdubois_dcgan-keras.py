

from __future__ import print_function, division

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import cv2

import pandas as pd

import numpy as np

import os

from tqdm import tqdm, tqdm_notebook

from keras.preprocessing.image import load_img

from keras.applications.densenet import preprocess_input

from keras.preprocessing.image import img_to_array

from keras.preprocessing.image import array_to_img

from keras.preprocessing import image

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.









img_size = 64

batch_size = 128







def resize_to_square(im):

    # new_size should be in (width, height) format

    im = cv2.resize(im, (img_size, img_size))

    return im



def load_image2(file):

    image = cv2.imread(file)

    new_image = resize_to_square(image)

    #ew_image = preprocess_input(new_image)

    return new_image



def load_image(file):

    new_image = load_img(file, target_size=(img_size, img_size))

    new_image = (img_to_array(new_image))

    #new_image = resize_to_square(img_to_array(new_image))

    #ew_image = preprocess_input(new_image)

    return new_image

    

if not os.path.exists('../output_images'):

    os.mkdir('../output_images')

    





from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise

from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D

from keras.layers import MaxPooling2D, concatenate

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam

from keras import losses

from keras.utils import to_categorical

import keras.backend as K



import matplotlib.pyplot as plt



import numpy as np







from pathlib import Path

from dataclasses import dataclass

import warnings



import numpy as np

from PIL import Image

from scipy import linalg

import tensorflow as tf



MODEL_PATH = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'

TRAIN_DIR = Path('../input/generative-dog-images/all-dogs/')

OUT_DIR = Path('../output_images/')

OUT_DIR.mkdir(exist_ok=True)





class KernelEvalException(Exception):

    pass





@dataclass

class MiFIDEvaluator(object):

    model_path: str

    train_images_path: str

    feature_path: str = None

    imsize: int = 64

    output_layer: str = 'Pretrained_Net/pool_3:0'

    input_layer: str = 'Pretrained_Net/ExpandDims:0'

    output_shape: int = 2048

    cosine_distance_eps: float = 0.1

    batch_size: int = 50

    fid_epsilon: float = 1e-14

    

    def __post_init__(self):

        tf.reset_default_graph()

        self.create_model_graph()

        with tf.Session() as sess:

            if self.feature_path is None:

                self.mu2, self.sigma2, self.features2 = self._handle_path_memorization(

                    self.train_images_path, sess, is_checksize=False, is_check_png=False)

            else:

                with np.load(self.feature_path) as f:

                    self.mu2, self.sigma2, self.features2 = f['m'], f['s'], f['features']

    

    def create_model_graph(self):

        with tf.gfile.FastGFile(self.model_path, 'rb') as f:

            graph_def = tf.GraphDef()

            graph_def.ParseFromString(f.read())

            _ = tf.import_graph_def(graph_def, name='Pretrained_Net')

            

    def img_read_checks(self, filename, is_checksize=False, is_check_png=False):

        im = Image.open(str(filename))

        if is_checksize and im.size != (self.imsize, self.imsize):

            raise KernelEvalException(f'The images are not of size {check_imsize}')

        if is_check_png and im.format != 'PNG':

            raise KernelEvalException('Only PNG images should be submitted.')



        if self.imsize is None:

            return im

        else:

            return im.resize((self.imsize, self.imsize), Image.ANTIALIAS)

        

    def _get_model_layer(self, sess):

        layer = sess.graph.get_tensor_by_name(self.output_layer)

        ops = layer.graph.get_operations()

        for op_idx, op in enumerate(ops):

            for o in op.outputs:

                shape = o.get_shape()

                if shape._dims != []:

                    shape = [s.value for s in shape]

                    new_shape = []

                    for j, s in enumerate(shape):

                        if s == 1 and j == 0:

                            new_shape.append(None)

                        else:

                            new_shape.append(s)

                    o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

        return layer

        

    def get_activations(self, images, sess):

        inception_layer = self._get_model_layer(sess)

        n_images = images.shape[0]

        if self.batch_size > n_images:

            warnings.warn('batch size is bigger than the data size. setting batch size to data size')

            self.batch_size = n_images

        n_batches = n_images // self.batch_size + 1

        pred_arr = np.empty((n_images, self.output_shape))

        for i in range(n_batches):

            start = i * self.batch_size

            if start + self.batch_size < n_images:

                end = start + self.batch_size

            else:

                end = n_images



            batch = images[start:end]

            pred = sess.run(inception_layer, {self.input_layer: batch})

            pred_arr[start:end] = pred.reshape(-1, self.output_shape)

        return pred_arr

        

    def calculate_activation_statistics(self, images, sess):

        act = self.get_activations(images, sess)

        mu = np.mean(act, axis=0)

        sigma = np.cov(act, rowvar=False)

        return mu, sigma, act

            

    def _handle_path_memorization(self, path, sess, is_checksize, is_check_png):

        path = Path(path)

        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))



        # In production we don't resize input images. This is just for demo purpose. 

        x = np.array([np.array(self.img_read_checks(fn, is_checksize, is_check_png)) for fn in files])

        m, s, features = self.calculate_activation_statistics(x, sess)

        del x

        return m, s, features

    

    def calculate_frechet_distance(self, mu1, sigma1):

        mu1 = np.atleast_1d(mu1)

        mu2 = np.atleast_1d(self.mu2)

        sigma1 = np.atleast_2d(sigma1)

        sigma2 = np.atleast_2d(self.sigma2)



        assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'

        assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'



        # product might be almost singular

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if not np.isfinite(covmean).all():

            msg = f'fid calculation produces singular product; adding {self.eps} to diagonal of cov estimates'

            warnings.warn(msg)

            offset = np.eye(sigma1.shape[0]) * self.eps

            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

            

        # numerical error might give slight imaginary component

        if np.iscomplexobj(covmean):

            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):

                m = np.max(np.abs(covmean.imag))

                raise ValueError(f'Imaginary component {m}')

            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (mu1 - mu2).dot(mu1 - mu2) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    

    @staticmethod

    def normalize_rows(x):

        return np.nan_to_num(x / np.linalg.norm(x, ord=2, axis=1, keepdims=True))

    

    def cosine_distance(self, features1):

        features1_nozero = features1[np.sum(features1, axis=1) != 0]

        features2_nozero = self.features2[np.sum(self.features2, axis=1) != 0]

        norm_f1 = self.normalize_rows(features1_nozero)

        norm_f2 = self.normalize_rows(features2_nozero)



        d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))

        mean_min_d = np.mean(np.min(d, axis=1))

        return mean_min_d

            

    def calculate_kid_given_paths(self, user_images_unzipped_path):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            m1, s1, features1 = self._handle_path_memorization(

                user_images_unzipped_path, sess, is_checksize=True, is_check_png=True)



            fid_value = self.calculate_frechet_distance(m1, s1)

            distance = self.cosine_distance(features1)

            return fid_value, distance

        

    def distance_thresholding(self, d):

        if d < self.cosine_distance_eps:

            return d

        else:

            return 1

        

    def evaluate(self, user_images_unzipped_path):

        fid_value, distance = self.calculate_kid_given_paths(user_images_unzipped_path)

        distance = self.distance_thresholding(distance)

        return fid_value, distance, fid_value / (distance + self.fid_epsilon)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()



train_generator = train_datagen.flow_from_directory(

        '../input/generative-dog-images/all-dogs/',

        target_size=(64, 64),

        batch_size=batch_size)



train_datagen_augment = ImageDataGenerator( featurewise_center=True,

        featurewise_std_normalization=True,

        rotation_range=20,

        horizontal_flip=True,

        shear_range=0.2)





train_generator_augment = train_datagen_augment.flow_from_directory(

        '../input/generative-dog-images/all-dogs/',

        target_size=(64, 64),

        batch_size=batch_size)
#plt.imshow(image.array_to_img(train_generator[0][0][0]))

#plt.show()



#plt.imshow(image.array_to_img(train_generator_augment[0][0][0]))

#plt.show()
class DCGAN():

    def __init__(self):

        # Input shape

        self.img_rows = 64

        self.img_cols = 64

        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 100



        optimizer = Adam(0.001, 0.5)



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

            optimizer=optimizer,

            metrics=['accuracy'])



        # Build the generator

        self.generator = self.build_generator()



        # The generator takes noise as input and generates imgs

        z = Input(shape=(self.latent_dim,))

        img = self.generator(z)



        # For the combined model we will only train the generator

        self.discriminator.trainable = False



        # The discriminator takes generated images as input and determines validity

        valid = self.discriminator(img)



        # The combined model  (stacked generator and discriminator)

        # Trains the generator to fool the discriminator

        self.combined = Model(z, valid)

        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)



    def build_generator(self):



        model = Sequential()



        model.add(Dense(128 * 16 * 16, activation="relu", input_dim=self.latent_dim))

        model.add(Reshape((16, 16, 128)))

        model.add(UpSampling2D())

        model.add(Conv2D(128, kernel_size=3, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))

        model.add(UpSampling2D())

        model.add(Conv2D(64, kernel_size=3, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))

        model.add(Activation("tanh"))



        model.summary()



        noise = Input(shape=(self.latent_dim,))

        img = model(noise)



        return Model(noise, img)



    def build_discriminator(self):



        model = Sequential()



        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))

        model.add(ZeroPadding2D(padding=((0,1),(0,1))))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))



        model.summary()



        img = Input(shape=self.img_shape)

        validity = model(img)



        return Model(img, validity)



    def train(self, epochs, batch_size=128, save_interval=50,times_augment=0):



        # Load the dataset

        increase_batch=times_augment+1



       # Adversarial ground truths

        valid = np.ones((batch_size*increase_batch, 1))

        fake = np.zeros((batch_size*increase_batch, 1))



        for epoch in range(epochs):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random half of images



            # Select a random batch of images and encode

            #idx = np.random.randint(0, len(train_generator)-2)

            imgs = train_generator[0][0]

            for i in range(times_augment):

                imgs_augment =train_generator_augment[0][0]

                imgs=np.concatenate((imgs,imgs_augment), axis=0)

            imgs = (imgs - 127.5) / 127.5



            # Sample noise and generate a batch of new images

            noise = np.random.normal(0, 1, (batch_size*increase_batch, self.latent_dim))

            gen_imgs = self.generator.predict(noise)



            # Train the discriminator (real classified as ones and generated as zeros)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------

            #  Train Generator

            # ---------------------



            # Train the generator (wants discriminator to mistake images as real)

            g_loss = self.combined.train_on_batch(noise, valid)



            # Plot the progress

            if (epoch % save_interval == 0) or (epoch==epochs):

                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



            # If at save interval => save generated image samples

            if (epoch % save_interval == 0) or (epoch==epochs):

                self.save_imgs(epoch)

                

            if (epoch % save_interval == 0) or (epoch==epochs):          

                for i in range(int(5)):

                    r, c = 5, 5

                    noise = np.random.normal(0, 1, (r * c, self.latent_dim))

                    gen_imgs = self.generator.predict(noise)



                    # Rescale images 0 - 1

                    gen_imgs = 0.5 * gen_imgs + 0.5



                    for j in range(20):

                        img = image.array_to_img((gen_imgs[j, :,:,:]))

                        img.save(os.path.join('../output_images/','generated_dog' + str(i) + '_'+ str(j) +'.png')) 

                plt.imshow(image.array_to_img(gen_imgs[3]))

                plt.show()

                evaluator = MiFIDEvaluator(MODEL_PATH, TRAIN_DIR / 'all-dogs/')

                fid_value, distance, mi_fid_score = evaluator.evaluate(OUT_DIR)

                print(f'FID: {fid_value:.5f}')

                print(f'distance: {distance:.5f}')

                print(f'MiFID: {mi_fid_score:.5f}')

        for i in range(int(5)):

            r, c = 5, 5

            noise = np.random.normal(0, 1, (r * c, self.latent_dim))

            gen_imgs = self.generator.predict(noise)



            # Rescale images 0 - 1

            gen_imgs = 0.5 * gen_imgs + 0.5



            for j in range(20):

                img = image.array_to_img((gen_imgs[j, :,:,:]))

                img.save(os.path.join('../output_images/','generated_dog' + str(i) + '_'+ str(j) +'.png')) 

            plt.imshow(image.array_to_img(gen_imgs[3]))

            plt.show()

            evaluator = MiFIDEvaluator(MODEL_PATH, TRAIN_DIR / 'all-dogs/')

            fid_value, distance, mi_fid_score = evaluator.evaluate(OUT_DIR)

            print(f'FID: {fid_value:.5f}')

            print(f'distance: {distance:.5f}')

            print(f'MiFID: {mi_fid_score:.5f}')

    def save_imgs(self, epoch):

        r, c = 5, 5

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))

        gen_imgs = self.generator.predict(noise)



        # Rescale images 0 - 1

        gen_imgs = 0.5 * gen_imgs + 0.5



        fig, axs = plt.subplots(r, c)

        cnt = 0

        for i in range(r):

            for j in range(c):

                axs[i,j].imshow(gen_imgs[cnt, :,:,])

                axs[i,j].axis('off')

                cnt += 1

        #fig.savefig("images/mnist_%d.png" % epoch)

        plt.show()

        plt.close()


if __name__ == '__main__':

    dcgan = DCGAN()

    dcgan.train(epochs=3000, batch_size=batch_size, save_interval=500,times_augment=0)



    

import shutil

shutil.make_archive('images', 'zip', '../output_images')


