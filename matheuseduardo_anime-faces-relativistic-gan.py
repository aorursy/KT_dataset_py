import numpy as np

import matplotlib.pyplot as plt

from PIL import Image as img

from os import listdir



from keras import Model, backend as K

from keras.layers import *

from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import Progbar
epochs = 30

batch_size = 64

alpha = 1 # Variable used to decay noise and rotation.

stddev = .2

rotation_range = 10

directory = '../input/data/data/'

files = len(listdir(directory))

real_shape = (64, 64)
class VariableGaussianNoise(Layer):

    def __init__(self, stddev, seed=None, **kwargs):

        self.stddev = K.variable(stddev)

        self.seed = seed

        super(VariableGaussianNoise, self).__init__(**kwargs)

    

    def call(self, x, training=None):

        def noise():

            return x+K.random_normal(K.shape(x), stddev=self.stddev, seed=self.seed)

            

        return K.in_train_phase(noise,

                                x, training=training)

    

    def compute_output_shape(self, input_shape):

        return input_shape

        

    def set_stddev(self, stddev):

        K.set_value(self.stddev, max(0., stddev))
def gen():

    x = Input((512,))

    y = Dense(4*4*512)(x)

    y = LeakyReLU(.2)(y)

    y = Reshape((4, 4, 512))(y)

    

    y = Conv2D(512, 3, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = UpSampling2D()(y)

    

    y = Conv2DTranspose(512, 3, strides=2, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(256, 3, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2DTranspose(256, 3, strides=2, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(128, 3, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2DTranspose(128, 3, strides=2, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(64, 3, padding='same')(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2D(3, 1, activation='tanh')(y)

    

    return Model(x, y)



def disc():

    x = Input(real_shape+(3,))

    y = VariableGaussianNoise(stddev, name='noise')(x)

    

    y = Conv2D(64, 3)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(128, 4, strides=2)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2D(128, 3)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(256, 4, strides=2)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2D(256, 3)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(512, 4, strides=2)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Conv2D(512, 3)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    y = Conv2D(512, 2)(y)

    y = LeakyReLU(.2)(y)

    y = BatchNormalization()(y)

    

    y = Flatten()(y)

    y = Dense(1024)(y)

    y = LeakyReLU(.2)(y)

    y = Dense(1)(y)

    

    return Model(x, y)
gen = gen()

disc = disc()



gen.summary()

disc.summary()



real = Input(real_shape+(3,))

z = Input((512,))

fake = gen(z)

disc_r = disc(real)

disc_f = disc(fake)



# Generator and discriminator losses according to RaGAN described in Jolicoeur-Martineau (2018).

# Credits to Smith42 on GitHub for the implementation. (https://github.com/Smith42/keras-relativistic-gan)

def relGenLoss(y_true, y_pred):

    epsilon=0.000001

    return -(K.mean(K.log(K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0)\

            +K.mean(K.log(1-K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0))



def relDiscLoss(y_true, y_pred):

    epsilon=0.000001

    return -(K.mean(K.log(K.sigmoid(disc_r - K.mean(disc_f, axis=0))+epsilon), axis=0)\

            +K.mean(K.log(1-K.sigmoid(disc_f - K.mean(disc_r, axis=0))+epsilon), axis=0))



gen.trainable = True

disc.trainable = False

gen_train = Model([real, z], [disc_r, disc_f])

gen_train.compile(SGD(2e-3, .5, nesterov=True), [relGenLoss, None])

gen_train.summary()



gen.trainable = False

disc.trainable = True

disc_train = Model([real, z], [disc_r, disc_f])

disc_train.compile(SGD(3e-3, .5, nesterov=True), [relDiscLoss, None])

disc_train.summary()
def dataGenerator():

    file_list = listdir(directory)

    image_data_generator = ImageDataGenerator(rotation_range = 10*alpha,

                                              horizontal_flip = True,

                                              preprocessing_function = lambda x: x/127.5-1)

    

    for i in range(files//batch_size):

        images = []

        

        for j in range(batch_size):

            images.append(np.array(img.open(directory+file_list[i*batch_size+j]).resize(real_shape)))

            

        yield next(image_data_generator.flow(np.array(images),

                                             batch_size=batch_size,

                                             shuffle=False))
dummy_y = np.zeros((batch_size,))

gen_loss = []

disc_loss = []



for e in range(epochs):

    target = files//batch_size

    progbar = Progbar(target)

    data_generator = dataGenerator()

    

    print('Epoch ', e+1, '/', epochs)

    for i in range(target):

        progbar.update(i+1)

        

        gen.trainable = False

        disc.trainable = True

        data = next(data_generator)

        z = np.random.normal(scale=.25, size=(batch_size, 512))

        disc_loss.append(disc_train.train_on_batch([data, z], dummy_y))

        

        gen.trainable = True

        disc.trainable = False

        gen_loss.append(gen_train.train_on_batch([data, z], dummy_y))

    

    alpha *= .9

    disc.get_layer('noise').set_stddev(alpha*stddev)

    

plt.plot(gen_loss, 'r-')

plt.plot(disc_loss, 'b-')

plt.savefig('Losses.png')



fig, axs = plt.subplots(5, 5, figsize=(10, 10))

for i in range(5):

    for j in range(5):

        axs[i][j].axis('off')

        axs[i][j].imshow(gen.predict(np.random.normal(scale=.25, size=(1, 512)))[0]/2+.5)

plt.savefig('Samples.png')



np.save('Generator Weights.npy', gen.get_weights())

np.save('Discriminator Weights.npy', disc.get_weights())