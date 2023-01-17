import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf
dir = "/kaggle/input/faces-data-new/images/"

def load_faces():

    data = []

    sizes = []

    for i in os.listdir(dir):   

        if '.jpg' in i:

            img = Image.open(dir + i)

            size = np.array(img,dtype = "float32").shape

            sizes.append(size)

            img = img.resize((120,120))

            # Convert to tf format   pixels = tf.keras.preprocessing.image.img_to_array(img)

            # Normalize data

            pixels = np.array(img,dtype = "float32")/255

            data.append(pixels)

        else:

            print("")

    return sizes, np.stack(data)

# Check image sizes and load data

sizes, dataset = load_faces()

print("Number of images:", len(sizes))

print("Unique Shapes of images:", pd.Series(sizes).unique() )

dataset.shape
plt.figure(figsize = (15,15))

for i in range(10):

    plt.subplot(5,5,i+1)

    plt.axis("off")

    plt.imshow(dataset[i])

plt.show()
 #RGB

# np.unique(dataset[2][:,:,1])



import numpy as np

green_back = [] 

for d in dataset:

    a = d[:,:,1]*255 #red

    b =d[:,:,0]*255 #blue

    if (pd.Series(a.flat).mean() > 70) & (pd.Series(a.flat).std() < 40):

    #pd.Series(a.flat).value_counts().index[0] > 60:

        green_back.append(d)

    else:

       "Not green"



len(green_back)
dataset = np.stack(green_back)


plt.figure(figsize = (15,15))

for i in range(50):

    plt.subplot(10,7,i+1)

    plt.axis("off")

    plt.imshow(dataset[i])

plt.show()
# CNN model 3*[Conv2D -> Leaky] -> Conv2D -> Dropout -> Flatten -> Dense + Sigmoid



def discriminator(inp_shape = (120,120,3)):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Conv2D(128, kernel_size = 3, strides = 2,  padding="same", 

               input_shape = inp_shape, kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.02) ),

        tf.keras.layers.LeakyReLU(0.2),

        

        tf.keras.layers.Conv2D(128, kernel_size = 3, strides = 2, padding="same"),

        tf.keras.layers.LeakyReLU(0.2),

        

        tf.keras.layers.Conv2D(64, kernel_size = 3, strides = 2, padding="same"),

        tf.keras.layers.LeakyReLU(0.2),

        

        tf.keras.layers.Conv2D(64, kernel_size = 3, strides = 2, padding = "same"),

        tf.keras.layers.Flatten(),      

        tf.keras.layers.Dense(1, activation = "sigmoid")

    ],

        name="discriminator")

    model.compile(optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5), loss = "binary_crossentropy", metrics = ['acc'])

    return model
# View model layers 

d_model = discriminator()

tf.keras.utils.plot_model(d_model, show_shapes = True)

# d_model.summary()
# Generative Model with BatchNormalization and LeakyRelu 

def generator(latent_dim = 100):

    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(128 * 15 * 15, input_dim = latent_dim,

                              kernel_initializer = tf.random_normal_initializer(mean=0, stddev=0.02) ),

        tf.keras.layers.LeakyReLU(0.2),

        tf.keras.layers.Reshape((15,15,128)),



        # First transpose convolutional filter 

        tf.keras.layers.Conv2DTranspose(128, kernel_size = 3, strides = 2,  padding = "same"),

#         tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(0.2),



        # Second transpose convolutional filter 

        tf.keras.layers.Conv2DTranspose(128, kernel_size = 3, strides = 2,  padding = "same"),

#         tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(0.2),



         # Third transpose convolutional filter 

        tf.keras.layers.Conv2DTranspose(64, kernel_size = 3, strides = 2,  padding = "same"),

#         tf.keras.layers.BatchNormalization(),

        tf.keras.layers.LeakyReLU(0.2),

        

        tf.keras.layers.Conv2D(3,  kernel_size = 3, padding = "same", activation = "tanh") #"sigmoid")

        

    ])

    return model
g_model = generator()

tf.keras.utils.plot_model(g_model, show_shapes = True)

# g_model.summary()
def gan(g_model, d_model):

    d_model.trainable = False

    model = tf.keras.models.Sequential([

        g_model,

        d_model

    ],

        name="DCGANs")

    model.compile(optimizer = tf.keras.optimizers.Adam(0.0002, beta_1=0.5), loss = "binary_crossentropy")

    return model
gan_model = gan(g_model, d_model)

tf.keras.utils.plot_model(gan_model, show_shapes = True)

# gan_model.summary()
def summarize_performance(g_model, dataset, n_size = 128):

    X_real, y_real = generate_real_samples(dataset)

    _,accr = d_model.evaluate(X_real, y_real)

    

    X_fake, y_fake = generate_fake_examples(g_model)

    _, accf = d_model.evaluate(X_fake, y_fake)

    

    print("Real samples Acc: {}".format(accr*100))

    print("Fake samples Acc: {}".format(accf*100))

    

#     plot_samples(X_fake)

    plt.figure(figsize = (15,15))

    for i in range(7*7):

        plt.subplot(7,7,i+1)

        plt.axis("off")

        plt.imshow(Xfake[i])

    plt.show()

    




def train(g_model, d_model, gan_model, dataset, iterations=2000, batch_size=200, latent_dim=100, sample_interval=200):

 losses = []

 accuracies = []

 # Labels for real and fake examples

 real = np.ones((batch_size, 1))

 fake = np.zeros((batch_size, 1))



 for iteration in range(iterations):



    # -------------------------

    #  Train the Discriminator

    # -------------------------

    # Select a random batch of real images

    ind = np.random.randint(0, dataset.shape[0], batch_size)

    imgs = dataset[ind]



    # Generate points from the latent space 

    z = np.random.normal(0, 1, (batch_size, latent_dim))

    # Generate a batch of fake images

    gen_imgs = g_model.predict(z)



    # Discriminator loss function 

    d_loss_real = d_model.train_on_batch(imgs, real) # real = 0 label

    d_loss_fake = d_model.train_on_batch(gen_imgs, fake) # fake = 1 label

    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



    # ---------------------

    #  Train the Generator

    # ---------------------

    # Generate point from a latent space 

    z = np.random.normal(0, 1, (batch_size, latent_dim))



    # GANs loss

    g_loss = gan_model.train_on_batch(z, real)



    # Generate a batch of fake images

    gen_imgs = g_model.predict(z)





    if iteration % sample_interval == 0:



    # Output training progress

        print ("%d [D loss: %f, Acc.: %.2f%%] [G loss: %f]" %

                  (iteration, d_loss[0], 100*d_loss[1], g_loss))



        # Save losses and accuracies to be plotted after training

        losses.append((d_loss[0], g_loss))

        accuracies.append(100*d_loss[1])



        # Output generated images

        #          summarize_performance(g_model, dataset)

        plt.figure(figsize = (15,15))

        for i in range(7*7):

            plt.subplot(7,7,i+1)

            plt.axis("off")

            plt.imshow(gen_imgs[i])

        plt.show()

train(g_model, d_model, gan_model, dataset)