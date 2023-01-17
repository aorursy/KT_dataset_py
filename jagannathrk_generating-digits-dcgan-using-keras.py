import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras import layers

from keras import initializers

from keras.optimizers import Adam
generator = Sequential()

generator.add(layers.Dense(256, input_shape=(100,), kernel_initializer=initializers.RandomNormal(stddev = 0.02)))

generator.add(layers.LeakyReLU(alpha=0.2))

generator.add(layers.Dense(512))

generator.add(layers.LeakyReLU(alpha=0.2))

generator.add(layers.Dense(1024))

generator.add(layers.LeakyReLU(alpha=0.2))

generator.add(layers.Dense(28*28, activation='tanh'))
optimizer = Adam(lr=0.0002, beta_1=0.5)

generator.compile(optimizer= optimizer, loss='binary_crossentropy')

generator.summary()
def generate_images(count=1):

    input = np.random.normal(0, 1, size=[count, 100])

    images = generator.predict(input)

    images *= 255 # pixels should be in range 0-255

    images = images.reshape((count, 28, 28))

    return images

    
image = generate_images(3)

plt.figure(figsize=(15,5))

for i in range(3):

    plt.subplot('13{0}'.format(i+1))

    plt.imshow(image[i], cmap='gray')

plt.show()
discriminator = Sequential()

discriminator.add(layers.Dense(1024, input_dim=784))

discriminator.add(layers.LeakyReLU(0.2))

discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Dense(512))

discriminator.add(layers.LeakyReLU(0.2))

discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Dense(256))

discriminator.add(layers.LeakyReLU(0.2))

discriminator.add(layers.Dropout(0.3))

discriminator.add(layers.Dense(1, activation='sigmoid'))

discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

discriminator.summary()
gan = Sequential()

gan.add(generator)

discriminator.trainable = False

gan.add(discriminator)

gan.compile(optimizer=optimizer, loss='binary_crossentropy')

gan.summary()
train_data = pd.read_csv('../input/digit-recognizer/train.csv')

samples = train_data.drop(columns='label').values

samples = samples/255 - 1
history = pd.DataFrame(columns=['d_loss', 'd_acc', 'gan_loss'])


def train(batch_size=32, epochs=1):

    half_batch_size = np.floor(batch_size/2).astype(np.int32)

    steps = int(len(samples)/batch_size)

    for epoch in range(epochs):

        for step in range(steps):

            y = np.zeros(batch_size) #fake_images

            y[:half_batch_size] = 0.9  # real images

            generated_images = generator.predict(np.random.normal(0, 1, size=[half_batch_size, 100]))

            real_images = samples[np.random.randint(0, samples.shape[0], size=half_batch_size)]

            x = np.concatenate((generated_images,real_images))

            discriminator.trainable = True

            d_metrics = discriminator.train_on_batch(x, y)

            noise = np.random.normal(0, 1, size=[batch_size, 100])

            discriminator.trainable = False

            gan_loss = gan.train_on_batch(noise, np.zeros(batch_size))

            history.loc[epoch*steps+steps] = [d_metrics[0], d_metrics[1], gan_loss]

        # visualize training progress

        str = f'Epoch {epoch}: [D loss: {d_metrics[0]} acc: {d_metrics[1]}] | [G loss: {gan_loss}'

        print(str)

        images = generate_images(9)

        plt.figure(figsize=(15,5))

        for i in range(9):

            plt.subplot('19{0}'.format(i+1))

            plt.imshow(images[i], cmap='gray_r' ,interpolation='nearest')

        plt.show()





    return history
history = train(batch_size=128, epochs=20)
plt.figure(figsize=(15, 5))

plt.title('Discriminator loss')

plt.plot(history['d_loss'])
plt.figure(figsize=(15, 5))

plt.title('GAN loss')

plt.plot(history['gan_loss'])
images = generate_images(9)

plt.figure(figsize=(15,5))

for i in range(9):

    plt.subplot('19{0}'.format(i+1))

    plt.imshow(images[i], cmap='gray_r' ,interpolation='nearest')

plt.savefig('generated_digits.png')