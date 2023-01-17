#Cree un nuevo kernel porque el otro me estaba dando MUCHOS problemas 16/11/2019

#Generative Adversarial Network (GAN)

import numpy as np

import operator

from functools import reduce

from keras.models import Sequential, Model

from keras.layers import * # Dense, Conv2D, Flatten, Dropout, LeakyRelu

from keras.optimizers import Adam, SGD

from keras.utils.vis_utils import plot_model

from keras_preprocessing.image import ImageDataGenerator

from keras.models import load_model

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import pathlib

import shutil

from imageio import imsave

import os





print(os.listdir("../input"))
dir_img = '../input/stanford-dogs-dataset/images/Images/'

dir_ann = "../input/stanford-dogs-dataset/annotations/Annotation/"

forma = (64, 64, 3) #forma de la imagen 64*64 pixeles en 3 canales

Batch = 32 #estaba entre 32 64 u 128, se eligio 32 para que el num batch sea significativo

latent_dim = 100 #numero de nodos usados en el input del generador, se eligio 100 para que no sea tan alto o tan bajo

epoch = 80 #tarda bastante, le voy a poner 2 para el commit pero idealmente queremos entre 80 y 150

#a mayor cantidad de epochs, mejor para el entrenamiento 

num_batch = 20579 // Batch // 2  #se baja el numero de batch

print(num_batch)
datagen = ImageDataGenerator( 

    horizontal_flip = True,

    rotation_range = 20,

    width_shift_range = 0.2,

    height_shift_range = 0.2,

    preprocessing_function = lambda X: (X-127.5)/127.5

) #Generar batches de tensor image data con aumentacion en tiempo real, la data se loopea

data_loader=datagen.flow_from_directory(

    dir_img,

    target_size = forma[:-1], 

    class_mode = None,

    batch_size = Batch)
def generar(n_pruebas=Batch):

    return next(data_loader)[:n_pruebas], np.ones((n_pruebas, 1))*0.9
def mostrar(ary, rows, cols):

    plt.figure(figsize=(cols*3, rows*3))

    for row in range(rows):

        for col in range(cols):

            plt.subplot(rows, cols, row*cols+col+1)

            img = (ary[row*cols+col, :] + 1) / 2

            plt.axis('off')

            plt.title(f'{row*cols+col}')

            plt.imshow(img)

    plt.show()
data, y = generar()

print('shape of data:', data.shape) # => (32, 64, 64, 3) forma de las imagenes

print('min, max of data:', data.min(), data.max()) # 0.0 1.0

print('shape of y', y.shape) # (32, 1)

print('min, max of y', y.min(), y.max()) # 1.0 1.0

print('head 5 of y', y[:5]) # [[1.] [1.] ...]



mostrar(data,2, 5)
#discriminador

def crear_discriminador():

    model = Sequential()

    model.add(Conv2D(32, kernel_size=4, strides=2, padding='same', input_shape=(64,64,3)))

    model.add(LeakyReLU(0.2))

    model.add(Conv2D(64, kernel_size=4, strides=2, padding='same'))

    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))

    model.add(BatchNormalization(momentum=0.1, epsilon=1e-05))

    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, kernel_size=4, strides=2, padding='same'))

    model.add(BatchNormalization(momentum=0.1, epsilon=1e-05))

    model.add(LeakyReLU(0.2))

    model.add(Conv2D(1, kernel_size=4, strides=1, padding='same'))

    model.add(Flatten())

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0005, beta_1=0.5))

    return model





discriminator = crear_discriminador()

discriminator.summary()
#generador con tanh preprocesado -1 a 1

def crear_generador():

    struct_ = (64, 8, 8)

    n_nodes = reduce(operator.mul, struct_) 

    print(f'dim generador input={n_nodes}')

    model = Sequential([

        Dense(n_nodes, activation='relu', input_shape=(latent_dim,)),

        Reshape((*struct_[1:], struct_[0])),

        BatchNormalization(momentum=0.8),

        

        #subir a 16*16

        UpSampling2D(),

        Conv2D(struct_[0], kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        #a 32*32

        UpSampling2D(),

        Conv2D(struct_[0]//2, kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        #a 64*64

        UpSampling2D(),

        Conv2D(struct_[0]//4, kernel_size=3, padding='same'),

        Activation('relu'),

        BatchNormalization(momentum=0.8),

        Conv2D(3, kernel_size=3, padding='same'),

        Activation('tanh'),

    ])



    return model





generator = crear_generador()

generator.summary()
# puntos latentes para generador

def generar_puntos(latent_dim, n_pruebas):

    ruido = np.random.uniform(-1, 1, (n_pruebas, latent_dim))

    return ruido
# generar falsas

def generar_falsas(g_model, latent_dim, n_pruebas):

    x_input = generar_puntos(latent_dim, n_pruebas)

    #predecir output

    X = g_model.predict(x_input)

    y = np.zeros((n_pruebas, 1))

    return X, y
# prueba output

X, y = generar_falsas(generator, latent_dim, 8)

mostrar(X, 2, 4)
#GAN

def crear_gan(generator, discriminator):

    discriminator_fijo = Model(inputs=discriminator.inputs, outputs=discriminator.outputs)

    discriminator_fijo.trainable = False

    model = Sequential([InputLayer(input_shape=(latent_dim,)), generator, discriminator_fijo])

    opt = Adam(lr=0.0005, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt)

    return model



gan = crear_gan(generator, discriminator)

gan.summary()
# Entrenamiento de los modelos

def ent_discriminador():

    #reales random

    X_real, y_real = generar(Batch//2)

    loss_real = discriminator.train_on_batch(X_real, y_real)

    #falsas

    X_fake, y_fake = generar_falsas(generator, latent_dim, Batch//2)

    loss_fake = discriminator.train_on_batch(X_fake, y_fake)

    return (loss_real+loss_fake)*0.5



def ent_gan(num_loop=1):

    X = generar_puntos(latent_dim, Batch)

    y = np.ones((Batch, 1))*0.9

    for i in range(num_loop):

        loss = gan.train_on_batch(X, y)

    return loss
# Entrenarlos a todos

history = np.zeros((epoch, num_batch, 2))

perrosenEpoch = np.zeros((epoch, *forma))



for i in tqdm(range(epoch), desc='epoch'):

    data_loader.reset()

    pbar_batch = tqdm(range(num_batch), desc='batch')

    

    for j in pbar_batch:

        d_loss = ent_discriminador()

        g_loss = ent_gan()

        pbar_batch.set_description(f'{i:>2}, d_loss:{d_loss:.2}, g_loss:{g_loss:.2}')

        history[i, j, :] = d_loss, g_loss

        

    generadas = generar_falsas(generator, latent_dim, 5)[0]

    mostrar(generadas, 1, 5)

    perrosenEpoch[i, :] = generadas[0,:]
# generando

latent_points = generar_puntos(latent_dim, 10000)

X = generator.predict(latent_points)

print(X.shape, X[0].min(), X[0].max())



mostrar(X, 2, 4)
#output

imgs = [((img+1) * 127.5).astype(np.uint8) for img in X]

np.array(imgs).min(), np.array(imgs).max()

imgdir = pathlib.Path('images')

if not imgdir.exists():

    imgdir.mkdir()



for n in range(len(imgs)):

    imsave(imgdir/f'doggo_{n}.png', imgs[n])

    

shutil.make_archive('images', 'zip', 'images')
#Como son muchos objetos para un commit, borrar

!rm -rf images

!ls