import numpy as np

import matplotlib.pyplot as plt



import keras

from keras import layers

from keras.layers import Dense, Dropout, Input, LeakyReLU, BatchNormalization, Conv2D

from keras.models import Model,Sequential

from keras.datasets import fashion_mnist

from tqdm import tqdm

from keras.layers.advanced_activations import LeakyReLU

from keras.optimizers import Adam

from keras import initializers



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
#can be used if discriminator first layer is Dense

def load_data(first_layer="dense"):

    (x_train, y_train), (_, _) = fashion_mnist.load_data()

    x_train = (x_train.astype(np.float32) - 127.5)/127.5  #normalize the values between -1 and 1

    print(x_train.shape)

    # convert shape of x_train from (60000, 28, 28) to (60000, 784) - 784 columns per row

    if first_layer == "dense":

        X_train = x_train.reshape(60000, 784)

    elif first_layer == "conv":

        # Select class 6 images (class 6)

#         x_train = x_train[y_train.flatten() == 6]

        print((x_train.shape[0],) + (height, width, channels))

        X_train = x_train.reshape((x_train.shape[0],) + (height, width, channels))

    return X_train



X_train = load_data("conv")

print(X_train.shape)
height = 28

width = 28

channels = 1 #gray scale image

latent_dim = 100  #shape of noise vector

epochs=20

batch_size=128

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
def create_generator(latent_dim, optimizer):

    generator=Sequential()

    generator.add(Dense(units=256, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))  #initialize weights using RandomNormal

#     generator.add(Dense(units=256, input_dim=input_dim))  #initialize weights using RandomNormal

    generator.add(LeakyReLU(0.2))

    

    generator.add(Dense(units=512))

    generator.add(LeakyReLU(0.2))

    

    generator.add(Dense(units=1024))

    generator.add(LeakyReLU(0.2))

    

    generator.add(Dense(units=784, activation='tanh'))

    

    #Adam optimizer as it is computationally efficient and has very little memory requirement. Adam is a combination of Adagrad and RMSprop !

    generator.compile(loss='binary_crossentropy', optimizer= optimizer)

    return generator



def create_conv_generator(latent_dim):

    nch = 200

    generator_input = Input(shape=(latent_dim,)) #Input(shape=[latent_dim]) #keras.Input(shape=(latent_dim,))   

    x = layers.Dense(nch*14*14, init='glorot_normal')(generator_input) 

    x = layers.BatchNormalization()(x) 

    x = layers.Activation('relu')(x)

    x = layers.Reshape( (14, 14, nch) )(x)  #14*14*200

    x = layers.UpSampling2D(size=(2, 2))(x) #28*28*200

        

    # Few more conv layers

    x = layers.Conv2D(100, (3, 3), padding='same')(x) #28*28*100

    x = layers.BatchNormalization()(x)  #28*28*100

    x = layers.LeakyReLU()(x)  #28*28*100

    

    x = layers.Conv2D(50, (3, 3), padding='same')(x) #28*28*50

    x = layers.BatchNormalization()(x)  #28*28*50

    x = layers.LeakyReLU()(x)  #28*28*50

       

    # Produce a 28x28 1-channel feature map

    x = layers.Conv2D(channels, (1, 1), activation='sigmoid', padding='same')(x)  #28*28*1

    generator = keras.models.Model(generator_input, x)

    return generator



generator = create_conv_generator(latent_dim) #create_generator(latent_dim, keras.optimizers.adam(lr=0.0002, beta_1=0.5))

generator.summary()
def create_discriminator(input_dim, optimizer):

    discriminator=Sequential()

    discriminator.add(Dense(units=1024, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02))) #initialize weights using RandomNormal    

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.3))

           

    discriminator.add(Dense(units=512))

    discriminator.add(LeakyReLU(0.2))

    discriminator.add(Dropout(0.3))

       

    discriminator.add(Dense(units=256))

    discriminator.add(LeakyReLU(0.2))

#     discriminator.add(Dropout(0.3))

    

    discriminator.add(Dense(units=1, activation='sigmoid'))

    

    discriminator.compile(loss='binary_crossentropy',optimizer= optimizer)    

    

    return discriminator



def create_conv_discriminator(height, width, channels):

    discriminator_input = layers.Input(shape=(height, width, channels))

    x = layers.Conv2D(256, (5, 5), subsample=(2, 2), padding = 'same', activation='relu')(discriminator_input) 

    x = layers.LeakyReLU(0.2)(x) 

    x = layers.Dropout(0.4)(x)   

    

    x = layers.Conv2D(512, (5, 5), subsample=(2, 2), padding='same')(x)

    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dropout(0.4)(x) 

    x = layers.Flatten()(x)



    x = layers.Dense(256)(x)

    x = layers.LeakyReLU(0.2)(x)

    x = layers.Dropout(0.4)(x)



    # Two class classification layer

    x = layers.Dense(1, activation='sigmoid')(x)



    discriminator = keras.models.Model(discriminator_input, x)



    # To stabilize training, we use learning rate decay and gradient clipping (by value) in the optimizer.

    discriminator.compile(optimizer=keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8), loss='binary_crossentropy')

    return discriminator



discriminator = create_conv_discriminator(height, width, channels) #create_discriminator(784, keras.optimizers.adam(lr=0.0002, beta_1=0.5))

discriminator.summary()
#Stacking The Generator And Discriminator Networks To form a GAN

def create_gan(generator, discriminator, latent_dim, optimizer): 

    #Setting the trainable parameter of discriminator to False. This will lock the discriminator and will not train it.  

    discriminator.trainable=False

    

    #Instantiates a keras tensor with shape of the noise vector

    gan_input = Input(shape=(latent_dim,))

    

    #Feeds the input noise to the generator and stores the output in Z

    z = generator(gan_input)

    

    #Feeds the output from generator(Z) to the discriminator and stores the result in out

    gan_output= discriminator(z)  #the output of discriminator is 1 or 0

    

    #Creates a model include all layers required in the computation of out given inp.

    gan= Model(inputs=gan_input, outputs=gan_output)  #100>784>1

    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gan



gan = create_gan(generator, discriminator, latent_dim, keras.optimizers.adam(lr=0.0002, beta_1=0.5))

gan.summary()
def plot_generated_images(epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):

    #Generate a normally distributed noise of shape(100(examples)x100)

    noise= np.random.normal(loc=0, scale=1, size=[examples, latent_dim])

    

    #Generate an image for the input noise

    generated_images = generator.predict(noise)

    

    #Reshape the generated image 

    generated_images = generated_images.reshape(examples,28,28)

    #Plot the image 

    plt.figure(figsize=figsize)

    

    #Plot for each pixel

    for i in range(generated_images.shape[0]):

        plt.subplot(dim[0], dim[1], i+1)

        plt.imshow(generated_images[i], cmap='gray_r', interpolation='nearest')

        plt.axis('off')

    plt.tight_layout()

    plt.savefig('gan_generated_image_epoch_%d.png' %epoch)

    plt.show()
def train(gan, discriminator, generator, X_train, latent_dim, epochs=1, batch_size=128):       

    batch_count = int(X_train.shape[0] / batch_size)

    print('Epochs: ', epochs)

    print('Batch size: ', batch_size)

    print('Batches per epoch: ', batch_count)

    printed=False

    

    for epoch in range(1,epochs+1 ):

        print('_'*15, "Epoch %d" %epoch, '_'*15)

        for _ in range(batch_size):

            #get a random set of input noises and images. Sample random points in the latent space

            noise = np.random.normal(0,1, size=(batch_size, latent_dim)) #generate random noise as an input to initialize the generator. Shape: 128(batch_size) * 100                       

            image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)] # Get a random set of real images from the training set. Shape: 128 * 784

                        

            # Generate fake MNIST images from noised input          

            generated_images = generator.predict(noise)          #Shape: 128 * 784

                            

            #Construct different batches of real and fake data for input to discriminator. Concatenate to create new training set with real and fake images 

            X = np.concatenate([image_batch, generated_images])   #shape: 256 * 784  -> top 128 images are real images while bottom 128 images are fake ones

            

            # Labels for generated(fake) and real data for input to discriminator

            y_dis=np.zeros(2*batch_size)  #256 where everything is 0

            y_dis[:batch_size]=0.9  # labeling every real images as 0.9 probability (not setting as 1 which would be very stringent and hard for the generator)

                                    # first 128 values for real images will have values 0.9 while generated images will have values 0

            if not printed: 

                print('noise.shape:', noise.shape)

                print('image_batch.shape:', image_batch.shape)

                print('generated_images.shape: ', generated_images.shape)

                print('X.shape', X.shape)

                print('y_dis.shape:', y_dis.shape)

                printed = True

            

            #Train the discriminator on fake and real data before starting the GAN. First we have to train the discriminator and then GAN sequentially on the loss of the discriminator

            discriminator.trainable=True

            d_loss = discriminator.train_on_batch(X, y_dis)

            

            # sample random points in the latent space

            noise= np.random.normal(0,1, size=(batch_size, latent_dim))  #Tricking the noised input of the Generator as real data

            y_gen = np.ones(batch_size)   #tell the discriminator that for all the input noises, label is 1 meaning all these are real images                 

            discriminator.trainable=False   # During the training of GAN, the weights of discriminator should be fixed. We can enforce that by setting the trainable flag

            

            #Train the GAN - will also train the generator now meaning updates of weights in generator will happen

            #Train the generator (via the gan model,where the discriminator weights are frozen)

            #training the GAN by alternating the training of the Discriminator and training the chained GAN model with Discriminator’s weights freezed.

            a_loss = gan.train_on_batch(noise, y_gen)

            

        #Plotting the images for every few epochs

        if epoch == 1 or epoch % 10 == 0:           

            plot_generated_images(epoch, generator)

            # Print metrics

            print('discriminator loss at epoch %s: %s' % (epoch, d_loss))

            print('adversarial loss at epoch %s: %s' % (epoch, a_loss))



train(gan, discriminator, generator, X_train, latent_dim, epochs=epochs, batch_size=batch_size)