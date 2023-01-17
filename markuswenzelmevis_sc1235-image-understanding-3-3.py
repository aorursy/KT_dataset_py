'''
DCGAN on MNIST using Keras
Code basis: Rowel Atienza; https://github.com/roatienza/Deep-Learning-Experiments
Adapted by Markus Wenzel, Fraunhofer MEVIS
'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        # On local (faster to read):
        # self.x_train = self.read_mnist_tf()
        # On Kaggle (also works locally, but slower):
        self.x_train = self.read_mnist()

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()
        
    def read_mnist(self):
        import pandas as pd
        mnist_train_df = pd.read_csv('../input/mnist-original/mnist_train.csv', sep=',', nrows=60000, header=None)
        train = mnist_train_df.values.astype(np.float32)
        x_train = train[5000:,1:].reshape(55000,28,28,1)
        return x_train/255. # Make it similar as TF data.
    
    def read_mnist_tf(self):
        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        return self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)


    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # [mwenzel] note that the discriminator loss is not fixed -- both the generator and the discriminator are optimized!
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=1000, batch_size=256, save_interval=100)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)
# Solution proposal

class DCGAN_new(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        assert self.img_rows == self.img_cols, "Input image has to be square, at least for now..."
        self.dims     = int(img_rows / 4)
        assert self.dims == img_rows / 4, "Input image extents have to be divisible by 4."
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.2
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.2
        depth = 64+64+64+64
        # In: 100
        # Out: dims x dims x depth
        self.G.add(Dense(self.dims*self.dims*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((self.dims, self.dims, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0001, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.00005, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

# Solution proposal

from os import listdir
from os.path import isfile, join, abspath
import numpy as np


class liver_DCGAN(object):
    def __init__(self):
        self.img_rows = 76
        self.img_cols = 76
        self.channel = 1
        
        ORIGINAL_PATH = './singleSlicePNGs/originals/'
        MASK_PATH = './singleSlicePNGs/masks/'

        original_files = sorted([join(ORIGINAL_PATH,f) for f in listdir(ORIGINAL_PATH) if (isfile(join(ORIGINAL_PATH, f)) and f.split('.')[1]=='png')])
        mask_files     = sorted([join(MASK_PATH,f)     for f in listdir(MASK_PATH)     if (isfile(join(MASK_PATH, f))     and f.split('.')[1]=='png')])

        permuted_file_indexer = np.random.permutation(np.arange(len(original_files)))
        crop_size = 76
        self.x_train = np.array([np.array(plt.imread(original_files[permuted_file_indexer[index]])[:crop_size,:crop_size,]) for index in permuted_file_indexer])
        self.y_train = np.array([np.array(plt.imread(mask_files[permuted_file_indexer[index]])[:crop_size,:crop_size,]) for index in permuted_file_indexer])

        self.DCGAN = DCGAN_new(self.img_rows, self.img_cols)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :]
            images_train = np.reshape(images_train, images_train.shape + (1,))
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'liver.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "liver_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    liver_dcgan = liver_DCGAN()
    timer = ElapsedTimer()
    liver_dcgan.train(train_steps=500, batch_size=256, save_interval=50)
    timer.elapsed_time()
    liver_dcgan.plot_images(fake=True)
    liver_dcgan.plot_images(fake=False, save2file=True)
# Solution proposal
from keras.layers import UpSampling2D, MaxPool2D
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import Input, concatenate
from keras.models import Model, Sequential

class DCGAN_segmentation(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        assert self.img_rows == self.img_cols, "Input image has to be square, at least for now..."
        self.dims     = int(img_rows / 4)
        assert self.dims == img_rows / 4, "Input image extents have to be divisible by 4."
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model
        print("DCGAN_segmentation initialised.")

    def addConvBNSequential(self, model, filters=32, kernel_size=(3,3), batch_norm=True, activation='prelu', padding='same', kernel_regularizer=None):
        if batch_norm:
            model = BatchNormalization()(model)
        if activation == 'prelu':
            model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='linear', kernel_regularizer=kernel_regularizer)(model)
            model = PReLU()(model)
        elif activation == 'lrelu':
            model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation='linear', kernel_regularizer=kernel_regularizer)(model)
            model = LeakyReLU()(model)
        else:
            model = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation, kernel_regularizer=kernel_regularizer)(model)
        return model

    def get_batchnorm_unet(self, _filters=32, _filters_add=0, _kernel_size=(3,3), _padding='same', _activation='prelu', _kernel_regularizer=None, _final_layer_nonlinearity='sigmoid', _batch_norm=True):
        if self.G:
            return self.G

        input_layer = Input(shape=(self.img_rows,self.img_cols,1))

        x0 = self.addConvBNSequential(input_layer, filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x0 = self.addConvBNSequential(x0,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x1 = MaxPool2D()(x0)

        x1 = self.addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x1 = self.addConvBNSequential(x1,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x2 = MaxPool2D()(x1)

        x2 = self.addConvBNSequential(x2,          filters=_filters+2*_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x2 = self.addConvBNSequential(x2,          filters=_filters+2*_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x3 = UpSampling2D()(x2)

        x3 = concatenate([x1,x3])
        x3 = self.addConvBNSequential(x3,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x3 = self.addConvBNSequential(x3,          filters=_filters+_filters_add, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x4 = UpSampling2D()(x3)

        x4 = concatenate([x0,x4])
        x4 = self.addConvBNSequential(x4,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)
        x4 = self.addConvBNSequential(x4,          filters=_filters, kernel_size=_kernel_size, padding=_padding, activation=_activation, kernel_regularizer=_kernel_regularizer, batch_norm=_batch_norm)

        output_layer = Conv2D(1, kernel_size=(1,1), activation=_final_layer_nonlinearity)(x4)

        self.G = Model(input_layer, output_layer)
        self.G.summary()
        return self.G    
    
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.2
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    # The learning rate of discriminator and adversarial model need to be adjusted carefully.
    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.00005, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM
    
    def generator_model(self):
        if self.GM:
            return self.GM
        optimizer = "adam"
        self.GM = self.get_batchnorm_unet(_activation='relu', _batch_norm=True)
        self.GM.compile(loss=self.dice_loss(), optimizer=optimizer)
        return self.GM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.get_batchnorm_unet(_activation='relu', _batch_norm=True))
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM

# Solution proposal

from os import listdir
from os.path import isfile, join, abspath
import numpy as np


class segmentation_DCGAN(object):
    def __init__(self):
        self.img_rows = 76
        self.img_cols = 76
        self.channel = 1
        
        ORIGINAL_PATH = './singleSlicePNGs/originals/'
        MASK_PATH = './singleSlicePNGs/masks/'
        original_files = [join(ORIGINAL_PATH,f) for f in listdir(ORIGINAL_PATH) if (isfile(join(ORIGINAL_PATH, f)) and f.split('.')[1]=='png')]
        mask_files     = [join(MASK_PATH,f)     for f in listdir(MASK_PATH)     if (isfile(join(MASK_PATH, f))     and f.split('.')[1]=='png')]
        original_files = sorted(original_files)
        mask_files     = sorted(mask_files)

        permuted_file_indexer = np.random.permutation(np.arange(len(original_files)))
        crop_size = 76
        self.x_train = np.array([np.array(plt.imread(original_files[permuted_file_indexer[index]])[:crop_size,:crop_size,]) for index in permuted_file_indexer])
        self.y_train = np.array([np.array(plt.imread(mask_files[permuted_file_indexer[index]])[:crop_size,:crop_size,]) for index in permuted_file_indexer])

        self.DCGAN = DCGAN_segmentation(self.img_rows, self.img_cols)
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.get_batchnorm_unet(_activation='relu', _batch_norm=True)

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            # Select a number of test images to predict
            rand_selection = np.random.randint(0, self.y_train.shape[0], size=16)
            originals_input = self.x_train[rand_selection, :, :]
            originals_input = np.reshape(originals_input, originals_input.shape + (1,))
        for i in range(train_steps):
            # images_train becomes the masks.
            # noise (the generator input) becomes the originals.
            # images_fake are the generator results -- the predicted masks.
            
            # (I) Train discriminator
            rand_selection = np.random.randint(0, self.x_train.shape[0], size=batch_size)
            originals_train = self.x_train[rand_selection, :, :]
            originals_train = np.reshape(originals_train, originals_train.shape + (1,))
            masks_train = self.y_train[rand_selection, :, :]
            masks_train = np.reshape(masks_train, masks_train.shape + (1,))
            masks_fake = self.generator.predict(originals_train)
            x = np.concatenate((masks_train, masks_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)
            
            # (II) Train generator
            

            # (III) Train adversarial
            rand_selection = np.random.randint(0, self.y_train.shape[0], size=batch_size)
            y = np.ones([batch_size, 1])
            originals_train = self.x_train[rand_selection, :, :]
            originals_train = np.reshape(originals_train, originals_train.shape + (1,))
            a_loss = self.adversarial.train_on_batch(originals_train, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=originals_input.shape[0],\
                        noise=originals_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'segmentation.png'
        if fake:
            if noise is None:
                # Replace noise with random input images
                rand_selection = np.random.randint(0, self.x_train.shape[0], size=samples)
                originals_train = self.x_train[rand_selection, :, :]
                noise = np.reshape(originals_train, originals_train.shape + (1,))
            else:
                filename = "segmentation_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]
            images = np.reshape(images, images.shape + (1,))

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    liver_dcgan = segmentation_DCGAN()
    timer = ElapsedTimer()
    liver_dcgan.train(train_steps=10000, batch_size=512, save_interval=50)
    timer.elapsed_time()
    liver_dcgan.plot_images(fake=True)
    liver_dcgan.plot_images(fake=False, save2file=True)
