from __future__ import print_function, division



from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Bidirectional, LSTM, Reshape, RepeatVector, TimeDistributed

from keras.layers import BatchNormalization, Activation

from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import UpSampling2D, Conv2D

from keras.models import Sequential, Model

from keras.optimizers import Adam



import matplotlib.pyplot as plt



import sys



import numpy as np



import os



from PIL import Image
def load_data():

    x_train = np.load(r'../input/blues-genre-midi-melodies/answers.npy',allow_pickle=True)

    x_train = x_train.reshape(721,4,4)

    return x_train
class LSTMGAN():

    def __init__(self):

        # Input shape

        self.img_rows = 4

        self.img_cols = 4

        self.img_shape = (self.img_rows, self.img_cols)

        self.latent_dim = 16



        optimizer = Adam(0.0001, 0.4)



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',

            optimizer=optimizer,

            metrics=['accuracy'])



        # Build the generator

        self.generator = self.build_generator()



        # The generator takes noise as input and generates song

        z = Input(shape=(4,4))

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

        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(4, 4)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Bidirectional(LSTM(128, return_sequences=True)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Bidirectional(LSTM(128)))

        model.add(LeakyReLU(alpha=0.2))

        #specifying output to have 40 timesteps

        model.add(RepeatVector(16))

        #specifying 1 feature as the output

        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.3))   

        model.add(TimeDistributed(Dense(128)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.4))

        model.add(TimeDistributed(Dense(128)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.4))

        model.add(TimeDistributed(Dense(1)))

        model.add(LeakyReLU(alpha=0.2))

        model.summary()



        noise = Input(shape=(4,4))

        img = model(noise)



        return Model(noise, img)



    def build_discriminator(self):



        model = Sequential()



        model.add(Bidirectional(LSTM(128, activation = 'relu', return_sequences=True), input_shape=(16, 1)))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Bidirectional(LSTM(128, activation = 'relu')))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.4))

        model.add(RepeatVector(1))

        model.add(TimeDistributed(Dense(128, activation = 'sigmoid')))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.4))

        model.add(TimeDistributed(Dense(128, activation = 'relu')))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.4))

        model.add(TimeDistributed(Dense(1, activation = 'linear')))

        model.summary()



        img = Input(shape=(16,1))

        validity = model(img)



        return Model(img, validity)

    



    def train(self, epochs, batch_size=128, save_interval=50):



        # Load the dataset

        X_train = load_data()



        # Rescale 0 to 1

        X_train = X_train / 128



        # Adversarial ground truths

        valid = np.ones((batch_size,1,1))

        fake = np.zeros((batch_size,1,1))



        for epoch in range(epochs):



            # ---------------------

            #  Train Discriminator

            # ---------------------



            # Select a random half of songs

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]

            imgs = np.array(imgs)

            imgs = imgs.reshape(len(imgs),16,1)



            # Sample noise and generate a batch of new songs

            noise = np.random.normal(0, 1, (batch_size,4,4))

            gen_imgs = self.generator.predict(noise)



            # Train the discriminator (real classified as ones and generated as zeros)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------

            #  Train Generator

            # ---------------------



            # Train the generator (wants discriminator to mistake songs as real)

            g_loss = self.combined.train_on_batch(noise, valid)



            # Plot the progress

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



            # If at save interval => save model

            if epoch % save_interval == 0:

                self.generator.save("LSTM_generator.h5")
lstmgan = LSTMGAN()

#lstmgan.train(epochs=1000, batch_size=20, save_interval=100)
from keras.models import load_model

model = load_model(r'../input/bilstm-gan/LSTM_generator.h5')
!pip install mido
import mido # easy to use python MIDI library

from mido import MidiFile, MidiTrack, Message
random = np.random.normal(0,1,(1,4,4))



predict = model.predict(random)



#adjusting for normalization

predict = predict * 128
print(predict)
midler = MidiFile()

track = MidiTrack()

midler.tracks.append(track)

track.append(Message('program_change', program=2, time=0))

for x in range(16):

    track.append(Message('note_on', note=int(predict[0][x][0]), velocity=64, time=20))

    track.append(Message('note_off', note=int(predict[0][x][0]), velocity=64, time=20))

    midler.save('new_song.mid')