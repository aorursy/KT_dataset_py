from keras import backend as K



from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda

from keras.layers import BatchNormalization, Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.utils import to_categorical



import matplotlib.pyplot as plt

import numpy as np



import sys



%matplotlib inline

print('done importing libraries')
img_rows = 28

img_cols = 28

channels = 1

img_shape = (img_rows, img_cols, channels)

z_dim = 100

num_classes = 10



print('done defining variables')
def load_mnist_data():

    with np.load('../input/mnist.npz') as f:

        x_train, y_train = f['x_train'], f['y_train']

        x_test, y_test = f['x_test'], f['y_test']

    return (x_train, y_train), (x_test, y_test)



class Dataset:

    def __init__(self, num_labeled):

        self.num_labeled = num_labeled

        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist_data()

        

        def preprocess_images(x):

            x = (x.astype(np.float32) - 127.5) / 127.5

            print(x.shape)

            x = np.expand_dims(x, axis=3)

            print(x.shape)

            

            return x

        

        def preprocess_labels(y):

            return y.reshape(-1, 1)

        

        self.x_train = preprocess_images(self.x_train)

        self.x_test = preprocess_images(self.x_test)

        

        self.y_train = preprocess_labels(self.y_train)

        self.y_test = preprocess_labels(self.y_test)

        

    def batch_labeled(self, batch_size):

        idx = np.random.randint(0, self.num_labeled, size=batch_size)

        images = self.x_train[idx]

        labels = self.y_train[idx]

        

        return images, labels

    

    def batch_unlabeled(self, batch_size):

        idx = np.random.randint(self.num_labeled, self.x_train.shape[0], batch_size)

        

        return self.x_train[idx]

    

    def training_set(self):

        x_train = self.x_train[:self.num_labeled]

        y_train = self.y_train[:self.num_labeled]

        

        return x_train, y_train

    

    def test_set(self):

        

        return self.x_test, self.y_test



print('done defining the dataset class')
num_labeled = 100

dataset = Dataset(num_labeled)



print('done instantiating the dataset class')
def build_generator(z_dim):

    

    model = Sequential()

    

    model.add(Dense(256*7*7, input_dim=z_dim))

    model.add(Reshape((7, 7, 256)))

    

    # 7*7*256 -> 14*14*256

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    

    # 14*14*128 -> 14*14*64

    model.add(Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    

    # 14*14*64 => 28*28*1

    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    

    model.add(Activation('tanh'))

    

    z = Input(shape=(z_dim,))

    img = model(z)

    

    return Model(z, img)

def build_discriminator(img_shape):

    

    model = Sequential()

    

    # 28*28*1 => 14*14*32

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))

    model.add(LeakyReLU(alpha=0.01))

    

    # 14*14*32 => 7*7*64

    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    

    # 7*7*64 => 3*3*128

    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))

    

    model.add(Dropout(rate=0.5))

    model.add(Flatten())

    

    model.add(Dense(num_classes))

    

    return model

def build_discriminator_supervised(discriminator_net):

    

    model = Sequential()

    

    model.add(discriminator_net)

    model.add(Activation('softmax'))

    

    return model

def build_discriminator_unsupervised(discriminator_net):

    

    model = Sequential()

    

    model.add(discriminator_net)

    

    def predict(x):

        prediction = 1.0 - (1.0 / (K.sum(K.exp(x), axis=-1, keepdims=True) + 1.0))

        

        return prediction

    

    model.add(Lambda(predict))

    

    return model

disc = build_discriminator(img_shape)



sup_disc = build_discriminator_supervised(disc)

sup_disc.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())



unsup_disc = build_discriminator_unsupervised(disc)

unsup_disc.trainable = False

unsup_disc.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam())



gen = build_generator(z_dim)



def combined(generator, discriminator):

    

    model = Sequential()

    

    model.add(generator)

    model.add(discriminator)

    

    return model



sgan = combined(gen, unsup_disc)

sgan.compile(loss='binary_crossentropy', optimizer=Adam())

d_accuracies = []

d_losses = []



def train(iterations, batch_size, sample_interval):

    real = np.ones((batch_size, 1))

    fake = np.zeros((batch_size, 1))

    

    for iteration in range(iterations):

        imgs, labels = dataset.batch_labeled(batch_size)

        labels = to_categorical(labels, num_classes=num_classes)

        

        imgs_unlabeled = dataset.batch_unlabeled(batch_size)

        

        z = np.random.normal(0, 1, (batch_size, z_dim))

        gen_imgs = gen.predict(z)

        

        d_loss_supervised, accuracy = sup_disc.train_on_batch(imgs, labels)

        d_loss_real = unsup_disc.train_on_batch(imgs_unlabeled, real)

        d_loss_fake = unsup_disc.train_on_batch(gen_imgs, fake)

        

        d_loss_unsupervised = 0.5 * np.add(d_loss_real, d_loss_fake)

        

        z = np.random.normal(0, 1, (batch_size, z_dim))

        gen_imgs = gen.predict(z)

        

        g_loss = sgan.train_on_batch(z, real)

        

        d_losses.append(d_loss_supervised)

        d_accuracies.append(accuracy)

        

        if iteration % sample_interval == 0:

            print('{} [D loss supervised: {:.4f}, acc: {:.2f}]'.format(iteration, d_loss_supervised, 100 * accuracy))

iterations = 8000

batch_size = 32

sample_interval = 800



train(iterations, batch_size, sample_interval)

x, y = dataset.test_set()

y = to_categorical(y, num_classes)



_, accuracy = sup_disc.evaluate(x, y)

print('Test accuracty = {:.2f}%'.format(accuracy * 100))

base_disc = build_discriminator(img_shape)

mnist_classifier = build_discriminator_supervised(base_disc)

mnist_classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam())

for i in range(100):

    x, y = dataset.batch_labeled(batch_size)

    y = to_categorical(y, num_classes=num_classes)

    sup_loss, sup_acc = mnist_classifier.train_on_batch(x, y)

    

    if i % 20 == 0:

        print('iteration = {} / loss = {:.4f} / accuracy = {:.2f}'.format(i, sup_loss, sup_acc * 100))

x, y = dataset.test_set()

y = to_categorical(y, num_classes)



_, accuracy = mnist_classifier.evaluate(x, y)

print('Test accuracty = {:.2f}%'.format(accuracy * 100))