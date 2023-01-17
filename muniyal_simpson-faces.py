import keras

from keras.layers import *

from keras.models import Sequential, Model

from keras.layers.advanced_activations import LeakyReLU

from keras.datasets import mnist

from keras.optimizers import Adam

from keras.preprocessing import image



import numpy as np

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

!mkdir images

!mkdir model

!ls
all_images= os.listdir("../input/cropped/")
exclude_images = ["9746","9731","9717","9684","9637","9641","9642","9584","9541","9535",

"9250","9251","9252","9043","8593","8584","8052","8051","8008","7957",

"7958","7761","7762","9510","9307","4848","4791","4785","4465","2709",

"7724","7715","7309","7064","7011","6961","6962","6963","6960","6949",

"6662","6496","6409","6411","6406","6407","6170","6171","6172","5617",

"4363","4232","4086","4047","3894","3889","3493","3393","3362","2780",

"2710","2707","2708","2711","2712","2309","2056","1943","1760","1743",

"1702","1281","1272","772","736","737","691","684","314","242","191"]



exclude_images = [img+'.png' for img in exclude_images]
len(exclude_images)
print(len(all_images))

print(len(exclude_images))
all_images = [i for i in all_images if i not in exclude_images]

print(len(all_images))
X_train = []



for i in all_images:

    img = image.load_img("../input/cropped/"+i, target_size=(64,64))

    img = image.img_to_array(img)

    X_train.append(img)



X_train = np.array(X_train)
X_train.shape
X_train = (X_train.astype('float32') - 127.5)/127.5

print(X_train.min())

print(X_train.max())

print(X_train.shape)
TOTAL_EPOCHS = 50

BATCH_SIZE = 64

NO_OF_BATCHES = X_train.shape[0]//BATCH_SIZE

HALF_BATCH = 128

NOISE_DIM = 100     #Upsample this vector in 784 

adam = Adam(lr=2e-4,beta_1=0.5)
# Generator Model - Learnable Upsampling



generator = Sequential()

generator.add(Dense(4*4*512, input_shape=(NOISE_DIM,))) # Upsampling 100 noise vector to 4*4*512 dimn vector

generator.add(Reshape((4,4,512))) # reshape to 3D 

generator.add(LeakyReLU(0.2))

generator.add(BatchNormalization())





# From (4,4,512) to (8,8,256) 





# generator.add(UpSampling2D())

# generator.add(Conv2D(filters=64, kernel_size=(5,5), padding='same'))



generator.add(Conv2DTranspose(256, kernel_size=(5,5), padding='same', strides=(2,2)))

generator.add(LeakyReLU(0.2))

generator.add(BatchNormalization())





# From (8,8,256) to (16,16,128)

generator.add(Conv2DTranspose(128, kernel_size=(5,5), padding='same', strides=(2,2)))

generator.add(LeakyReLU(0.2))

generator.add(BatchNormalization())





# From  (16,16,128) to (32,32,64)

generator.add(Conv2DTranspose(64, kernel_size=(5,5), padding='same', strides=(2,2)))

generator.add(LeakyReLU(0.2))

generator.add(BatchNormalization())





# From (32,32,64) to (64,64,3)



# generator.add(UpSampling2D())

# generator.add(Conv2D(filters=1, kernel_size=(5,5), padding='same', activation='tanh'))

generator.add(Conv2DTranspose(3, kernel_size=(5,5), padding='same', strides=(2,2), activation="tanh"))





generator.compile(loss="binary_crossentropy", optimizer=adam)

generator.summary()
#  Discriminator Model



# Recieve an input image of  (64,64,3) and convert to (32,32,32)

discriminator = Sequential()

discriminator.add(Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", input_shape=(64,64,3) ))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())



# Convert  (32,32,32) to (16,16,64)

discriminator.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding="same" ))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())





# Convert  (16,16,64) to (8,8,128)

discriminator.add(Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same" ))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())





# Convert   (8,8,128) to (8,8,256)

discriminator.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding="same" ))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())





# Convert   (8,8,256) to (4,4,512)

discriminator.add(Conv2D(filters=512, kernel_size=(5,5), strides=(2,2), padding="same" ))

discriminator.add(LeakyReLU(0.2))

discriminator.add(BatchNormalization())









# Convert (4,4,512) to Flatten vector of dimn 8192

discriminator.add(Flatten())

discriminator.add(Dense(1, activation="sigmoid"))



discriminator.compile(loss="binary_crossentropy", optimizer= adam)

discriminator.summary()
#  Create GAN 



discriminator.trainable = False

gan_input = Input(shape=(NOISE_DIM,))

generate_img = generator(gan_input)

gan_output = discriminator(generate_img)





# Keras Functional API for combining both models



model = Model(inputs= gan_input, outputs = gan_output)

model.compile(loss="binary_crossentropy", optimizer=adam)

model.summary()
def save_images(epoch, samples=100):

    noise = np.random.normal(0, 1, size=(samples, NOISE_DIM))

    generated_img = generator.predict(noise)

    generated_img = generated_img.reshape(-1,64,64)



    plt.figure(figsize=(10,10))

    for i in range(samples):

        plt.subplot(10,10,i+1)

        plt.imshow(generated_img[i], interpolation='nearest')

        plt.axis("off")

  

    plt.tight_layout()

    plt.savefig('images/gan_output_epoch_{0}.png'.format(epoch+1))

    plt.show()
# Training GAN 



discriminator_losses = []

generator_losses = []



for epoch in range(TOTAL_EPOCHS):

    epoch_d_loss = 0.

    epoch_g_loss = 0.



    # Mini Batch SGD

    for batch in range(NO_OF_BATCHES):

    

        # Step 1 Train Discriminator

        # 50% Real Data + 50% Fake Data



        # Real Data X

        idx = np.random.randint(0,X_train.shape[0],HALF_BATCH)

        real_imgs = X_train[idx]



        # Fake Data X

        noise = np.random.normal(0,1, size=(HALF_BATCH,NOISE_DIM))

        fake_imgs = generator.predict(noise) # Forward Pass



        # Labels

        real_y = np.ones((HALF_BATCH,1))*0.9 #One side Label Smoothing for Discriminator

        fake_y = np.zeros((HALF_BATCH,1))





        # Train Discriminator



        d_loss_real = discriminator.train_on_batch(real_imgs, real_y)

        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)

        d_loss = 0.5*d_loss_real + 0.5*d_loss_fake



        epoch_d_loss += d_loss











        # Step 2 Train Generator

        noise = np.random.normal(0, 1, size=(BATCH_SIZE,NOISE_DIM))

        ground_truth_y = np.ones((BATCH_SIZE,1))

        g_loss = model.train_on_batch(noise, ground_truth_y)



        epoch_g_loss += g_loss

    

    

    

    print("Epoch %d Disc Loss %.4f Generator Loss %.4f" %((epoch+1),epoch_d_loss/NO_OF_BATCHES,epoch_g_loss/NO_OF_BATCHES))

    discriminator_losses.append(epoch_d_loss)

    generator_losses.append(epoch_g_loss)

  

    if(epoch+1)%10==0:

        generator.save("model/generator_{0}.h5".format(epoch+1))

        save_images(epoch)
plt.plot(discriminator_losses, label= "disc loss")

plt.plot(generator_losses, label="gen loss")

plt.legend()

plt.show()