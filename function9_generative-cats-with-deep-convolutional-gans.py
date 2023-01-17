from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model



import matplotlib.pyplot as plt



import sys



import numpy as np



import os



from PIL import Image
os.listdir('../input/cats-faces-64x64-for-generative-models/')
data = []

paths = []

for r, d, f in os.walk(r'../input/cats-faces-64x64-for-generative-models/cats'):

    for file in f:

        if '.jpg' in file:

            paths.append(os.path.join(r, file))



for path in paths:

    img = Image.open(path)

    x = np.array(img)

    data.append(x)



x_train = np.array(data)

x_train = x_train.reshape(len(data),64,64,3)
class DCGAN():

    def __init__(self):

        # Input shape

        self.img_rows = 64

        self.img_cols = 64

        self.channels = 3

        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 150



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

            optimizer='Adamax',

            metrics=['accuracy'])



        # Build the generator

        self.generator = self.build_generator()

        generator = self.generator



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

        self.combined.compile(loss='binary_crossentropy', optimizer='Adamax')



    def build_generator(self):



        model = Sequential()

        model.add(Dense(150 * 4 * 4, input_dim=self.latent_dim))

        model.add(Reshape((4, 4, 150)))

        model.add(UpSampling2D())

        model.add(Conv2D(150, kernel_size=(3,3), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())

        model.add(Conv2D(150, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())

        model.add(Conv2D(150, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling2D())

        model.add(Conv2D(150, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(150, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(150, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(3, kernel_size=(2,2), padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.summary()



        noise = Input(shape=(self.latent_dim,))

        img = model(noise)



        return Model(noise, img)



    def build_discriminator(self):



        model = Sequential()



        model.add(Conv2D(100, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.5))

        model.add(Conv2D(100, kernel_size=3, strides=2, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.5))

        model.add(Conv2D(100, kernel_size=3, strides=2, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.5))

        model.add(Conv2D(100, kernel_size=3, strides=1, padding="same"))

        model.add(BatchNormalization(momentum=0.8))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.5))

        model.add(Flatten())

        model.add(Dense(1, activation='sigmoid'))



        model.summary()



        img = Input(shape=self.img_shape)

        validity = model(img)



        return Model(img, validity)

    



    def train(self, epochs, batch_size=128, save_interval=50):



        # Load the dataset

        X_train = x_train



        # Normalize data to 0-1

        X_train = X_train / 255



        # Adversarial ground truths

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        

        g_loss_epochs = np.zeros((epochs, 1))

        d_loss_epochs = np.zeros((epochs, 1))



        for epoch in range(epochs):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random half of images

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]



            # Sample noise and generate a batch of new images

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

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

            

            #save loss history

            g_loss_epochs[epoch] = g_loss

            d_loss_epochs[epoch] = d_loss[0]



            # Plot the progress

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



            # If at save interval => save generated image samples

            if epoch % save_interval == 0:

                self.save()

                

        return g_loss_epochs, d_loss_epochs



    def save(self):

        self.generator.save("generator.h5")
dcgan = DCGAN()
#not training full 50,000 epochs on here

g_loss, d_loss = dcgan.train(epochs=1000, batch_size=100, save_interval=100)
plt.plot(g_loss)

plt.plot(d_loss)

plt.title('GAN Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Generator', 'Discriminator'], loc='upper left')

plt.show()