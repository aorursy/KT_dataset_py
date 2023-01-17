import numpy as np



%matplotlib inline

import matplotlib.pyplot as plt
from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, LeakyReLU, BatchNormalization

from keras.optimizers import Adam

from keras import initializers



import warnings

warnings.filterwarnings("ignore")
(X_train, y_train), (X_test, y_test) = mnist.load_data()



# We only concerned with 'X_train' data

# It can be rewritten as (X_train,_),(_,_) = mnist.load_data()



prev = X_train.shape
X_train = X_train.reshape(60000, 28*28)



print(prev)

print(X_train.shape)
# normalizing the inputs (-1, 1)



X_train = (X_train.astype('float32') / 255 - 0.5) * 2

# we have pixels from 0-255, dividing by 255 leads to normalize them in range 0-1

#(-0.5) * 2 shift it to (-1,1) for tanh activation
latent_dim = 100

# Latent dimensions are dimensions which we do not directly observe, but which we assume to exist (Hidden)

# We use this in reference of generator, it create images from latent dimension whichwe  do not directly observe, but assume to exist



# image dimension 28x28

img_dim = 784



init = initializers.RandomNormal(stddev=0.02)

# stddev = Standard deviation of the random values to generate



# The neural network needs to start with some weights and then iteratively update them to better values. 

# kernel_initializer is term for which statistical distribution or function to use for initialising the weights.

# Ref - https://datascience.stackexchange.com/questions/37378/what-are-kernel-initializers-and-what-is-their-significance



# Generator network



# sequential model simply allows us to stitch layers together

generator = Sequential()



# Input layer and hidden layer 1

generator.add(Dense(128, input_shape=(latent_dim,), kernel_initializer=init))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization(momentum=0.8))



# A dense layer is simply a fully connected layer of neurons.



# The LeakyReLU remove problem of "dying ReLU" and alpha is negative slope constant

# Deep dive - https://towardsdatascience.com/complete-guide-of-activation-functions-34076e95d044



# batchnormalization layer will transform inputs so they will have a mean of zero and a standard deviation of one.

# “momentum” in batch norm allows you to control how much of the statistics from the previous mini batch to include when the update is calculated

# Deep dive - https://machinelearningmastery.com/how-to-accelerate-learning-of-deep-neural-networks-with-batch-normalization/



# Hidden layer 2

generator.add(Dense(256))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization(momentum=0.8))



# Hidden layer 3

generator.add(Dense(512))

generator.add(LeakyReLU(alpha=0.2))

generator.add(BatchNormalization(momentum=0.8))



# Output layer 

generator.add(Dense(img_dim, activation='tanh'))
generator.summary()
discriminator = Sequential()



# Input layer and hidden layer 1

discriminator.add(Dense(512, input_shape=(img_dim,), kernel_initializer=init))

discriminator.add(LeakyReLU(alpha=0.2))



# Hidden layer 2

discriminator.add(Dense(256))

discriminator.add(LeakyReLU(alpha=0.2))



# Hidden layer 3

discriminator.add(Dense(128))

discriminator.add(LeakyReLU(alpha=0.2))



# Output layer

discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()
optimizer = Adam(lr=0.0002, beta_1=0.5)

discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



# Since we have to predict either fake or real (i.e. two classes)
discriminator.trainable = False



d_g = Sequential()

d_g.add(generator)

d_g.add(discriminator)

d_g.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
d_g.summary()
epochs = 100

batch_size = 64
real = np.ones(shape=(batch_size, 1)) #craete a "real" array with values = 1 and size = 100 

fake = np.zeros(shape=(batch_size, 1)) # craete a "fake" array with values = 0 and size = 100
d_loss = [] #discriminator loss

d_g_loss = [] #adversarial loss/generator loss
for e in range(epochs + 1):

    for i in range(len(X_train) // batch_size):

        

        # Train Discriminator weights

        discriminator.trainable = True

        

        # Real samples

                

        X_batch = X_train[i*batch_size:(i+1)*batch_size]

        # Defining size of batches per 64 in one batch

        

        d_loss_real = discriminator.train_on_batch(x=X_batch, y=real * (0.9))

        # train_on_batch (predefined keras function) - Runs a single gradient update on a single batch of data.

        # Pre train discriminator on  fake and real data  before starting the gan.

        # helps to check if our compiled models run fine on our real data as well as the noised data.

        

       

        # Fake Samples      

                

        z = np.random.normal(loc=0, scale=1, size=(batch_size, latent_dim))

        # generate random noise as an input to initialize generator

        

        X_fake = generator.predict_on_batch(z)

        # Generate fake MNIST images from noised input

        

        d_loss_fake = discriminator.train_on_batch(x=X_fake, y=fake)

        # train discriminator on fake images generated by generator and fake data (array of 0 values)

         

        # Discriminator loss.... well what's this ?

        # we only grabbed half the number of images that we specified with the real loss, 

        # we're take other half images from our generator for the other half of the batch:



        

        

        d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        

        # Train Generator weights

        

        discriminator.trainable = False

        # When we train the GAN we need to freeze the weights of the Discriminator. 

        # GAN is trained by alternating training of Discriminator and then training chained GAN model with Discriminator weights frozen

        

        # during training of gan weights of discriminator should be fixed We can enforce that by setting the trainable flag

        

        d_g_loss_batch = d_g.train_on_batch(x=z, y=real)

        # training the GAN by alternating training of Discriminator 

        # training the chained GAN model with Discriminator’s weights freezed

        

        # We'll now train the GAN with mislabeled generator outputs ([z=Noise] with [real i.e. 1]). 

        # That means we will generate images from noise and assign a label to one of them while training with the GAN

        

        # But Why ?

        

        # we are using the newly trained discriminator to improve generated output

        # GAN loss is going to describe the confusion of discriminator from generated outputs.



 # Rest is for visualization   

        print('epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % 

            (e + 1, epochs, i, len(X_train) // batch_size, d_loss_batch, d_g_loss_batch[0]),100*' ',end='\r')

    

    d_loss.append(d_loss_batch)

    d_g_loss.append(d_g_loss_batch[0])

    print('epoch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e + 1, epochs, d_loss[-1], d_g_loss[-1]), 100*' ')



    if e % 10 == 0:

        samples = 10

        x_fake = generator.predict(np.random.normal(loc=0, scale=1, size=(samples, latent_dim)))



        for k in range(samples):

            plt.subplot(2, 5, k+1)

            plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')

            plt.xticks([])

            plt.yticks([])



        plt.tight_layout()

        plt.show()
plt.plot(d_loss)

plt.plot(d_g_loss)

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Discriminator', 'Adversarial'], loc='center right')

plt.show()