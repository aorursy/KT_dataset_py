import numpy as np
import os

from glob import glob

import cv2



from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.optimizers import Adam



%matplotlib inline

import matplotlib.pyplot as plt
#PATH = os.path.abspath(os.path.join('..','input', 'flowers', 'flowers', 'rose'))

PATH = os.path.abspath(os.path.join('..','input', 'roseimages', 'roseimages'))

IMGS = glob(os.path.join(PATH, "*.jpg"))



print(len(IMGS)) # number of the rose images

print(IMGS[:10]) # rose images filenames
WIDTH = 28

HEIGHT = 28

DEPTH = 3
def procImages(images):

    processed_images = []

    

    # set depth

    depth = None

    if DEPTH == 1:

        depth = cv2.IMREAD_GRAYSCALE

    elif DEPTH == 3:

        depth = cv2.IMREAD_COLOR

    else:

        print('DEPTH must be set to 1 or to 3.')

        return None

    

    #resize images

    for img in images:

        base = os.path.basename(img)

        full_size_image = cv2.imread(img, depth)

        processed_images.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))

    processed_images = np.asarray(processed_images)

    

    # rescale images to [-1, 1]

    processed_images = np.divide(processed_images, 127.5) - 1



    return processed_images
processed_images = procImages(IMGS)

processed_images.shape
fig, axs = plt.subplots(5, 5)

count = 0

for i in range(5):

    for j in range(5):

        img = processed_images[count, :, :, :] * 127.5 + 127.5

        img = np.asarray(img, dtype=np.uint8)

        if DEPTH == 3:

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        axs[i, j].imshow(img)

        axs[i, j].axis('off')

        count += 1

plt.show()
# GAN parameters

LATENT_DIM = 100

G_LAYERS_DIM = [256, 512, 1024]

D_LAYERS_DIM = [1024, 512, 256]



BATCH_SIZE = 16

EPOCHS = 1000

LR = 0.0002

BETA_1 = 0.5
def buildGenerator(img_shape):



    def addLayer(model, dim):

        model.add(Dense(dim))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        

    model = Sequential()

    model.add(Dense(G_LAYERS_DIM[0], input_dim=LATENT_DIM))

    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization(momentum=0.8))

    

    for layer_dim in G_LAYERS_DIM[1:]:

        addLayer(model, layer_dim)

        

    model.add(Dense(np.prod(img_shape), activation='tanh'))

    model.add(Reshape(img_shape))



    model.summary()



    noise = Input(shape=(LATENT_DIM,))

    img = model(noise)



    return Model(noise, img)
#g = buildGenerator(processed_images.shape[1:])
def buildDiscriminator(img_shape):



    def addLayer(model, dim):

        model.add(Dense(dim))

        model.add(LeakyReLU(alpha=0.2))



    model = Sequential()

    model.add(Flatten(input_shape=img_shape))

    

    for layer_dim in D_LAYERS_DIM:

        addLayer(model, layer_dim)

        

    model.add(Dense(1, activation='sigmoid'))

    model.summary()



    img = Input(shape=img_shape)

    classification = model(img)



    return Model(img, classification)
#d = buildDiscriminator(processed_images.shape[1:])
def buildCombined(g, d):

    # fix d for training g in the combined model

    d.trainable = False



    # g gets z as input and outputs fake_img

    z = Input(shape=(LATENT_DIM,))

    fake_img = g(z)



    # gets the classification of the fake image

    gan_output = d(fake_img)



    # the combined model for training generator g to fool discriminator d

    model = Model(z, gan_output)

    model.summary()

    

    return model
def sampleImages(generator):

    rows, columns = 5, 5

    noise = np.random.normal(0, 1, (rows * columns, LATENT_DIM))

    generated_imgs = generator.predict(noise)



    fig, axs = plt.subplots(rows, columns)

    count = 0

    for i in range(rows):

        for j in range(columns):

            img = generated_imgs[count, :, :, :] * 127.5 + 127.5

            img = np.asarray(img, dtype=np.uint8)

            if DEPTH == 3:

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            axs[i, j].imshow(img)

            axs[i, j].axis('off')

            count += 1

    plt.show()
#sampleImages(g)
#instantiate the optimizer

optimizer = Adam(LR, BETA_1)
#build the discriminator

d = buildDiscriminator(processed_images.shape[1:])

d.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#build generator

g = buildGenerator(processed_images.shape[1:])

g.compile(loss='binary_crossentropy', optimizer=optimizer)
#build combined model

c = buildCombined(g, d)

c.compile(loss='binary_crossentropy', optimizer=optimizer)
#training

SAMPLE_INTERVAL = WARNING_INTERVAL = 100



YDis = np.zeros(2 * BATCH_SIZE)

YDis[:BATCH_SIZE] = .9 #Label smoothing



YGen = np.ones(BATCH_SIZE)



for epoch in range(EPOCHS):

    # get a batch of real images

    idx = np.random.randint(0, processed_images.shape[0], BATCH_SIZE)

    real_imgs = processed_images[idx]



    # generate a batch of fake images

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    fake_imgs = g.predict(noise)

    

    X = np.concatenate([real_imgs, fake_imgs])

    

    # Train discriminator

    d.trainable = True

    d_loss = d.train_on_batch(X, YDis)



    # Train the generator

    d.trainable = False

    #noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    g_loss = c.train_on_batch(noise, YGen)



    # Progress

    if (epoch+1) % WARNING_INTERVAL == 0 or epoch == 0:

        print ("%d [Discriminator Loss: %f, Acc.: %.2f%%] [Generator Loss: %f]" % (epoch, d_loss[0], 100. * d_loss[1], g_loss))



    # If at save interval => save generated image samples

    if (epoch+1) % SAMPLE_INTERVAL == 0 or epoch == 0:

        sampleImages(g)
def buildGeneratorDC(img_shape):

    model = Sequential()



    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=LATENT_DIM))

    model.add(Reshape((7, 7, 128)))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(64, kernel_size=3, padding="same"))

    model.add(BatchNormalization(momentum=0.8))

    model.add(Activation("relu"))

    model.add(Conv2D(DEPTH, kernel_size=3, padding="same"))

    model.add(Activation("tanh"))



    model.summary()



    noise = Input(shape=(LATENT_DIM,))

    img = model(noise)



    return Model(noise, img)
def buildDiscriminatorDC(img_shape):

    model = Sequential()



    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))

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



    img = Input(shape=img_shape)

    classification = model(img)



    return Model(img, classification)
#build the discriminator

dDC = buildDiscriminatorDC(processed_images.shape[1:])

dDC.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#build generator

gDC = buildGeneratorDC(processed_images.shape[1:])

gDC.compile(loss='binary_crossentropy', optimizer=optimizer)
#build combined model

cDC = buildCombined(gDC, dDC)

cDC.compile(loss='binary_crossentropy', optimizer=optimizer)
#training DC GAN

SAMPLE_INTERVAL = WARNING_INTERVAL = 100



YDis = np.zeros(2 * BATCH_SIZE)

YDis[:BATCH_SIZE] = .9 #Label smoothing



YGen = np.ones(BATCH_SIZE)



for epoch in range(EPOCHS):

    # get a batch of real images

    idx = np.random.randint(0, processed_images.shape[0], BATCH_SIZE)

    real_imgs = processed_images[idx]



    # generate a batch of fake images

    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    fake_imgs = gDC.predict(noise)

    

    X = np.concatenate([real_imgs, fake_imgs])

    

    # Train discriminator

    dDC.trainable = True

    for _ in range(5):

        d_loss = dDC.train_on_batch(X, YDis)



    # Train the generator

    dDC.trainable = False

    #noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))

    g_loss = cDC.train_on_batch(noise, YGen)



    # Progress

    if (epoch+1) % WARNING_INTERVAL == 0 or epoch == 0:

        print ("%d [Discriminator Loss: %f, Acc.: %.2f%%] [Generator Loss: %f]" % (epoch, d_loss[0], 100. * d_loss[1], g_loss))



    # If at save interval => save generated image samples

    if (epoch+1) % SAMPLE_INTERVAL == 0 or epoch == 0:

        sampleImages(gDC)