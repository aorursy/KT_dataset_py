import numpy as np

def generate_real_samples(n):

    '''generate n real samples with class labels'''

    x1 = np.random.rand(n) - 0.5 #generate a random number between [-0.5,0.5]

    x2 = x1**3        #generate outputs

    x1 = x1.reshape(n, 1)

    x2 = x2.reshape(n, 1)

    X = np.hstack((x1, x2))   #stack layers

    y = np.ones((n, 1))     #generate class label

    return X,y
from keras.models import Sequential

from keras.layers import Dense, LeakyReLU

from keras.utils import plot_model

import matplotlib.pyplot as plt



def define_discriminator(inputs = 2):

    ''' function to return the compiled discriminator model'''

    model = Sequential()

    model.add(Dense(25, activation = 'relu', kernel_initializer = 'he_uniform', input_dim = inputs))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(15, activation = 'relu', kernel_initializer = 'he_uniform'))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(5, activation = 'relu', kernel_initializer = 'he_uniform'))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    return model



discriminator_model = define_discriminator()

discriminator_model.summary()

plot_model(discriminator_model, to_file = 'discriminator_model.png', show_shapes = True, show_layer_names = True)
def define_generator(latent_dim, outputs = 2):

    model = Sequential()

    model.add(Dense(25, activation = 'relu', kernel_initializer= 'he_uniform', input_dim = latent_dim))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(15, activation = 'relu', kernel_initializer = 'he_uniform'))

    model.add(LeakyReLU(alpha = 0.01))

    model.add(Dense(outputs, activation = 'linear'))

    return model
latent_dim = 5

generator_model = define_generator(latent_dim)

generator_model.summary()

plot_model(generator_model, to_file = 'generator_model.png', show_shapes = True, show_layer_names = True)
def generate_latent_points(latent_dim, n):

    '''generate points in latent space as input for the generator'''

    x_input = np.random.rand(latent_dim*n) #generate points in latent space

    x_input = x_input.reshape(n,latent_dim)  #reshape

    return x_input



def generate_fake_samples(generator, latent_dim, n):

    x_input = generate_latent_points(latent_dim, n) #genarate points in latent space

    x = generator.predict(x_input) #predict outputs

    y = np.zeros((n, 1))

    return x, y
X, _ = generate_fake_samples(generator_model, latent_dim, 100)

plt.scatter(X[:,0], X[:,1])

plt.show()
def define_gan(generator, discriminator):

    '''define the combined generator and discriminator model'''

    discriminator.trainable = False

    model = Sequential()

    model.add(generator)

    model.add(discriminator)

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy')

    return model
gan_model = define_gan(generator_model, discriminator_model)

gan_model.summary()

plot_model(gan_model, to_file = 'gan_model.png', show_layer_names = True, show_shapes = True)
def train_gan(g_model,d_model,gan_model,latent_dim, num_epochs = 10000,num_eval = 2000, batch_size = 128):

    ''' function to train gan model'''

    half_batch = int(batch_size/2)

  #run epochs

    for i in range(num_epochs):

        X_real, y_real = generate_real_samples(half_batch) #generate real examples

        d_model.train_on_batch(X_real, y_real)               # train on real data

        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch) #generate fake samples

        d_model.train_on_batch(X_fake, y_fake)                #train on fake data

        #prepare points in latent space as input for the generator

        x_gan = generate_latent_points(latent_dim, batch_size)

        y_gan = np.ones((batch_size, 1))    #generate fake labels for gan

        gan_model.train_on_batch(x_gan, y_gan)

        if (i+1) % num_eval == 0:

            summarize_performance(i + 1, g_model, d_model, latent_dim)
def summarize_performance(epoch, generator, discriminator, latent_dim, n = 100):

    '''evaluate the discriminator and plot real and fake samples'''

    x_real, y_real = generate_real_samples(n)      #generate real samples

    _, acc_real = discriminator.evaluate(x_real, y_real, verbose = 1)

    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)

    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose = 1)

    print('Epoch: ' + str(epoch) + ' Real Acc.: ' + str(acc_real) + ' Fake Acc.: '+ str(acc_fake))

    plt.scatter(x_real[:,0], x_real[:,1], color = 'red')

    plt.scatter(x_fake[:,0], x_fake[:,1], color = 'blue')

    plt.show()
train_gan(generator_model, discriminator_model, gan_model, latent_dim)