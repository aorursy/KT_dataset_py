from keras.layers import Dense, Dropout, ReLU, LeakyReLU, BatchNormalization

from keras.models import Sequential, Model, Input

from keras.optimizers import Adam

from keras.datasets import mnist

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
import os

print(os.listdir("../input"))
x_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

x_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")



x_train = np.array(x_train)

x_test = np.array(x_test)



x_train = x_train.astype("float32")/255.0

x_test = x_test.astype("float32")/255.0



x_train = x_train[:,:-1]

x_test = x_test[:,:-1]



print(x_train.shape)

print(x_test.shape)
def create_generator():

    generator = Sequential()

    generator.add(Dense(512, input_dim=100))

    generator.add(ReLU())



    generator.add(Dense(1024))

    generator.add(ReLU())



    generator.add(Dense(512))

    generator.add(ReLU())



    #çıktı boyutlarını verimizle uyumlu olacak şekilde 784 olarak ayarlayalım

    generator.add(Dense(784, activation="tanh"))





    #sahte ve gerçek iki sınıf olacağı için classification yapar gibi modelimizi kuracağız.

    generator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, beta_1 = 0.5))

    return generator



g = create_generator()

g.summary()
def create_discriminator():

    discriminator = Sequential()

    discriminator.add(Dense(1024, input_dim=784))

    discriminator.add(ReLU())

    discriminator.add(Dropout(0.4))

    

    discriminator.add(Dense(512))

    discriminator.add(ReLU())

    discriminator.add(Dropout(0.4))



    discriminator.add(Dense(512))

    discriminator.add(ReLU())



    discriminator.add(Dense(1, activation="sigmoid"))



    discriminator.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, beta_1=0.5))



    return discriminator



d = create_discriminator()

d.summary()
def create_gan(generator, discriminator):

    discriminator.trainable = False



    #şimdi bir input belirleyip generator'a vermekle başlayalım.

    gan_input = Input(shape=(100,))

    #generator bu kısmı aldıktan sonra bize bir değer verecek.

    x = generator(gan_input)

    #aldığımız değer discriminator'a girip denetlenecek ve bize bir GAN outputu dönderecek

    gan_output = discriminator(x)



    #artık bir gan modeli kurabiliriz.

    gan = Model(inputs=gan_input, outputs=gan_output)

    gan.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])



    #sonuç olarak gan modelimizi döndürelim

    return gan



gan = create_gan(g, d)

g.summary()
epochs = 50

batch_size = 256

acc_list = []

hsitory = []

def gan_train(g, d, gan):

    for e in range(epochs):

        print("Devam eden epoch : ", e+1)

        for _ in range(batch_size):

            noise = np.random.normal(0,1,[batch_size, 100])

            generated_img = g.predict(noise)

            

            batch_img = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

            

            x = np.concatenate([batch_img, generated_img])





            y_disc = np.zeros(batch_size*2)

            y_disc[:batch_size] = 1



            d.trainable = True

            d.train_on_batch(x, y_disc)



            noise = np.random.normal(0,1,[batch_size, 100])

            y_gen = np.ones(batch_size)



            d.trainable = False



            history = gan.train_on_batch(noise, y_gen)

        acc_list.append(history[0])

        history.append(history)

 

    print("Eğitim Tamamlandı...")



gan_train(g,d,gan)
noise = np.random.normal(loc=0, scale=1, size=[100,100])

generated_images = g.predict(noise)

generated_images = generated_images.reshape(100,28,28)

plt.imshow(generated_images[66], interpolation="nearest")

plt.title("GAN Sonucu\nÇıkan Resim")

plt.show()

gan