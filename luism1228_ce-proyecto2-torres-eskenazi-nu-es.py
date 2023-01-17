import numpy as np

import os

import shutil

import matplotlib.pyplot as plt

import zipfile

import tensorflow as tf

import xml.etree.ElementTree as ET

from tqdm import tqdm

from keras.models import Model 

from keras.models import Sequential

from keras.layers.core import Dense

from keras.layers.core import Dropout

from keras.layers import Input

from keras.layers import BatchNormalization

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Reshape

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import UpSampling2D

from keras.layers import ReLU

from keras.layers.advanced_activations import LeakyReLU

from keras.initializers import RandomNormal

from keras.optimizers import Adam

from keras import backend as K

from PIL import Image
# Constantes y directorios

SEED = 4250

np.random.seed(SEED)

random_dim = 128

ROOT_DIR = '../input/'

IMAGES_DIR = ROOT_DIR + 'images/Images/'

BREEDS_DIR = ROOT_DIR + 'annotations/Annotation/'

BREEDS = os.listdir(BREEDS_DIR)

IMAGES = []

FOLDERS=[]



'''for r, d, f in os.walk(IMAGES_DIR):

    for folder in d:

        os.listdir(os.path.join(r, folder)))'''

        

# r=root, d=directories, f = files

for r, d, f in os.walk(IMAGES_DIR):

    for file in f:

        if '.jpg' in file:

            IMAGES.append(os.path.join(file))



for r, d, f in os.walk(IMAGES_DIR):

    for folder in d:

        FOLDERS.append(os.path.join(r, folder))

# Resumen

print('Total Images: {}'.format(len(IMAGES)))

print('Total Annotations: {}'.format(len(BREEDS)))

print('Total Carpetas de perros: {}'.format(len(FOLDERS)))

        



def load_images():

    # Place holder for output 

    all_images = np.zeros((22250, 64, 64, 3))



    # Indice

    index = 0

    errores=[]

    for breed in BREEDS:

        for dog in os.listdir(BREEDS_DIR + breed):

            try: 

                img=Image.open(IMAGES_DIR+ breed+'/'+dog+'.jpg')

                #print(IMAGES_DIR+ breed+'/'+dog+'.jpg')

            except: continue

            tree = ET.parse(BREEDS_DIR + breed + '/' + dog)

            root = tree.getroot()

            objects = root.findall('object')



            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)



                # Determina cada lado



                xdelta = xmax - xmin

                ydelta = ymax - ymin

                #Filtrar imagenes donde la caja que delimita al perro es muy baja para ser usada

                if xdelta >= 64 and ydelta >= 64:

                    img2 = img.crop((xmin, ymin, xmax, ymax))

                    img2 = img2.resize((64, 64), Image.ANTIALIAS)

                    image = np.asarray(img2)



                    # Normaliza al rango [-1, 1]

                    if np.size(image, 2) == 3:

                        all_images[index,:] = (image.astype(np.float32) - 127.5)/127.5

                        index += 1

                    else:

                        errores.append(IMAGES_DIR+ breed+'/'+dog+'.jpg')







                if index % 1000 == 0:

                    print('Processed Images: {}'.format(index))

    print('Total Processed Images: {}'.format(index))

    print(errores)

    return all_images
def create_generator_model():

    #Inicializacion normal aleatoria de pesos

    init = RandomNormal(mean = 0.0, stddev = 0.02)



    # Modelo

    model = Sequential()



    # Comienza en 4 * 4

    start_shape = 64 * 4 * 4

    model.add(Dense(start_shape, kernel_initializer = init, input_dim = random_dim))

    model.add(Reshape((4, 4, 64)))

    

    # Upsample => 8 * 8 

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))

    model.add(ReLU())

    

    # Upsample => 16 * 16 

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))

    model.add(ReLU())

    

    # Upsample => 32 * 32

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))

    model.add(ReLU())

    

    # Upsample => 64 * 64

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size = 3, padding = "same", kernel_initializer = init))

    model.add(ReLU())

    

    # output

    model.add(Conv2D(3, kernel_size = 3, activation = 'tanh', padding = 'same', kernel_initializer=init))

    model.compile(loss = 'binary_crossentropy', optimizer='rmsprop')

    print(model.summary())



    return model
def create_discriminator_model():

    input_shape = (64, 64, 3)



    #Inicializacion normal aleatoria de pesos

    init = RandomNormal(mean = 0.0, stddev = 0.02)



    # Definir el modelo

    model = Sequential()



    # Downsample ==> 32 * 32

    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init, input_shape = input_shape))

    model.add(LeakyReLU(0.2))

    model.add(Dropout(0.25))



    # Downsample ==> 16 * 16

    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))

    model.add(LeakyReLU(0.2))

    model.add(Dropout(0.25))

    

    # Downsample => 8 * 8

    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))

    model.add(LeakyReLU(0.2))

    model.add(Dropout(0.25))

    

    # Downsample => 4 * 4

    model.add(Conv2D(128, kernel_size = 3, strides = 2, padding = 'same', kernel_initializer = init))

    model.add(LeakyReLU(0.2))

    model.add(Dropout(0.25))

    

    # Capas finales

    model.add(Flatten())

    model.add(Dense(1, activation = 'sigmoid', kernel_initializer = init))



    # Compila el modelo

    model.compile(loss = 'binary_crossentropy',optimizer='rmsprop')

    

    print(model.summary())

    

    return model
def create_gan_model(discriminator, random_dim, generator):

    # Setea trainable to False inicialmente

    discriminator.trainable = False

    

    # Input de la GAN

    gan_input = Input(shape = (random_dim,))

    

    # Generator Output...una imagen

    generator_output = generator(gan_input)

    

    # El output del discriminador es la probabilidad de que una imagen sea real o falsa

    gan_output = discriminator(generator_output)

    gan_model = Model(inputs = gan_input, outputs = gan_output)

    gan_model.compile(loss = 'binary_crossentropy', optimizer='rmsprop')

    print(gan_model.summary())

    

    return gan_model
def generator_input(latent_dim, n_samples):

    # Generar puntos en el espacio latente

    input = np.random.randn(latent_dim * n_samples)



    # Reshape del input batch para la red

    input = input.reshape((n_samples, latent_dim))



    return input
def plot_generated_images(epoch, generator, examples = 25, dim = (5, 5)):

    generated_images = generator.predict(np.random.normal(0, 1, size = [examples, random_dim]))

    generated_images = ((generated_images + 1) * 127.5).astype('uint8')

        

    plt.figure(figsize = (12, 8))

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0], dim[1], i + 1)

        plt.imshow(generated_images[i], interpolation = 'nearest')

        plt.axis('off')

    plt.suptitle('Epoch %d' % epoch, x = 0.5, y = 1.0)

    plt.tight_layout()

    plt.savefig('dog_at_epoch_%d.png' % epoch)

    

def plot_loss(d_f, d_r, g):

    plt.figure(figsize = (18, 12))

    plt.plot(d_f, label = 'Discriminator Fake Loss')

    plt.plot(d_r, label = 'Discriminator Real Loss')

    plt.plot(g, label = 'Generator Loss')

    plt.legend()

    plt.savefig('loss_plot.png')

    plt.close()
def train_model(epochs = 1, batch_size = 128):

    # Cargar Imagenes

    x_train = load_images()

    

    # Calcular el numero de batches

    batch_count = x_train.shape[0] / batch_size



    # Crear modelos del discriminador y generador

    generator = create_generator_model()

    discriminator = create_discriminator_model()

    

    # Crear Modelos GAN

    gan_model = create_gan_model(discriminator, random_dim, generator)

    

    # Listado para Loss History

    discriminator_fake_hist, discriminator_real_hist, generator_hist = [], [], []

    

    for e in range(epochs):

        

        # Script Stop Counter

        script_stopper_counter = 0

        

        print('======================== Epoch {} ============================='.format(e))

        for _ in tqdm(range(int(batch_count))):

            

            #Perdida del discriminador

            discriminator_fake_loss, discriminator_real_loss = [], []

            

            #Entrenar al discriminador mas que el generador

            for _ in range(2):

                # Entrenar el discriminador en imagenes falsas

                X_fake = generator.predict(generator_input(random_dim, batch_size))

                y_fake = np.zeros(batch_size)

                y_fake[:] = 0

                discriminator.trainable = True

                d_fake_loss = discriminator.train_on_batch(X_fake, y_fake)

                

                # Entrenar el discriminador en imagenes reales

                X_real = x_train[np.random.randint(0, x_train.shape[0], size = batch_size)]

                y_real = np.zeros(batch_size)

                y_real[:] = 0.9  # label smoothing

                discriminator.trainable = True

                d_real_loss = discriminator.train_on_batch(X_real, y_real)



                # Guardar Perdida en cada iteracion

                discriminator_fake_loss.append(d_fake_loss)

                discriminator_real_loss.append(d_real_loss)



            # Entrenar generador

            noise = generator_input(random_dim, batch_size)

            y_gen = np.ones(batch_size)

            discriminator.trainable = False

            generator_loss = gan_model.train_on_batch(noise, y_gen)





            # Guardar Loss en la lista de historial del Loss

            discriminator_fake_hist.append(np.mean(discriminator_fake_loss))

            discriminator_real_hist.append(np.mean(discriminator_real_loss)) 

            generator_hist.append(generator_loss)

            

            

            # En ocasiones el Loss del discriminador explota y se mantiene alto, en ese caso se para el script

            if np.mean(discriminator_fake_loss) > 10:

                script_stopper_counter += 1

        

        # Resume la calidad de la imagen por epochs durante el entrenamiento

        if e % 100 == 0:

            plot_generated_images(e, generator)

            

        #Parar script si un epoch tiene perdida explosiva

        if script_stopper_counter > 160:

            plot_generated_images(e, generator)

            break

            

    # Grafica la perdida durante el entrenamiento

    plot_loss(discriminator_fake_hist, discriminator_real_hist, generator_hist)



    # Crea Images.zip

    z = zipfile.PyZipFile('images.zip', mode = 'w')

    for k in range(1000):

        # Genera nuevos perros

        generated_images = generator.predict(np.random.normal(0, 1, size = [1, random_dim]))

        image = Image.fromarray(((generated_images + 1) * 127.5).astype('uint8').reshape(64, 64, 3))



        # Salvar en un .zip file 

        f = str(k)+'.png'

        image.save(f, 'PNG')

        z.write(f)

        os.remove(f)

        

        # Plot Status Counter

        if k % 100 == 0: 

            print(k)

    z.close()
train_model(300,256)