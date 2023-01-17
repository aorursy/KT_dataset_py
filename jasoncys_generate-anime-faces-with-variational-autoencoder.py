#IMPORTS



import numpy as np

import os



import matplotlib.pyplot as plt

%matplotlib inline



from tqdm import tqdm

import cv2
#LOADING THE DATA INTO A NUMPY ARRAY



x_data = []

for png in tqdm(os.listdir('../input/data/data')):

    path = '../input/data/data/{}'.format(png)

    

    image = plt.imread(path)

    image = image.astype('float32')

    x_data.append(image)

    

x_data = np.array(x_data)    
# KERAS IMPORTS



import keras



from keras.models import Sequential, Model, Input



from keras.layers import Dense

from keras.layers import Conv2D

from keras.layers import Conv2DTranspose

from keras.layers import MaxPool2D, AvgPool2D

from keras.layers import UpSampling2D

from keras.layers.advanced_activations import LeakyReLU



from keras.layers import Activation

from keras.layers import BatchNormalization

from keras.layers import Lambda



from keras.layers import Flatten

from keras.layers import Reshape



from keras.layers import Add, Multiply



from keras.losses import mse, binary_crossentropy



import keras.backend as K
#SET A SEED FOR REPRODUCABILITY

np.random.seed(20)



#NUMBER OF DIMENSIONS IN THE ENCODED LAYER

latent_dims = 512
#ENCODER

#BUILT WITH FUNCTIONAL MODEL DUE TO THE MULTIPLE INPUTS AND OUTPUTS



encoder_in = Input(shape=(64,64,3))   ##INPUT FOR THE IMAGE



encoder_l1 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', input_shape=(64,64,3))(encoder_in)

encoder_l1 = BatchNormalization()(encoder_l1)

encoder_l1 = Activation(LeakyReLU(0.2))(encoder_l1)



encoder_l1 = Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(encoder_l1)

encoder_l1 = BatchNormalization()(encoder_l1)

encoder_l1 = Activation(LeakyReLU(0.2))(encoder_l1)





encoder_l2 = Conv2D(filters=128, kernel_size=5, strides=2, padding='same')(encoder_l1)

encoder_l2 = BatchNormalization()(encoder_l2)

encoder_l2 = Activation(LeakyReLU(0.2))(encoder_l2)



encoder_l3 = Conv2D(filters=256, kernel_size=5, strides=2, padding='same')(encoder_l2)

encoder_l3 = BatchNormalization()(encoder_l3)

encoder_l3 = Activation(LeakyReLU(0.2))(encoder_l3)





encoder_l4 = Conv2D(filters=512, kernel_size=5, strides=2, padding='same')(encoder_l3)

encoder_l4 = BatchNormalization()(encoder_l4)

encoder_l4 = Activation(LeakyReLU(0.2))(encoder_l4)



flatten = Flatten()(encoder_l4)



encoder_dense = Dense(1024)(flatten)

encoder_dense = BatchNormalization()(encoder_dense)

encoder_out = Activation(LeakyReLU(0.2))(encoder_dense)





mu = Dense(latent_dims)(encoder_out)

log_var = Dense(latent_dims)(encoder_out)





epsilon = Input(tensor=K.random_normal(shape=(K.shape(mu)[0], latent_dims)))  ##INPUT EPSILON FOR RANDOM SAMPLING



sigma = Lambda(lambda x: K.exp(0.5 * x))(log_var) # CHANGE log_var INTO STANDARD DEVIATION(sigma)

z_eps = Multiply()([sigma, epsilon])



z = Add()([mu, z_eps])



encoder=Model([encoder_in,epsilon], z)

encoder.summary()

# DECODER

# BUILT WITH SEQUENTIAL MODEL AS NO BRANCHING IS REQUIRED



decoder = Sequential()

decoder.add(Dense(1024, input_shape=(latent_dims,)))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))



decoder.add(Dense(8192))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))



decoder.add(Reshape(target_shape=(4,4,512)))



decoder.add(Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same'))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))



decoder.add(Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))



decoder.add(Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))





decoder.add(Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same'))

decoder.add(BatchNormalization())

decoder.add(Activation(LeakyReLU(0.2)))



decoder.add(Conv2DTranspose(filters=3, kernel_size=5, strides=1, padding='same'))

decoder.add(BatchNormalization())

decoder.add(Activation('sigmoid'))



decoder.summary()

# COMBINE ENCODER AND DECODER THE COMPLETE THE VARIATIONAL AUTO ENCODER



vae_preds = decoder(z)

vae = Model([encoder_in, epsilon], vae_preds)



vae.summary()

# MY LOSS FUNCTIONS



def reconstruction_loss(y_true, y_pred):

    return K.mean(K.square(y_true - y_pred))



def kl_loss(y_true, y_pred):

    kl_loss = - 0.5 * K.mean(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)

    return kl_loss



def vae_loss(y_true, y_pred):

    return reconstruction_loss(y_true, y_pred) + 0.03 * kl_loss(y_true, y_pred)   #scaling kl_loss by 0.03 seem to help

vae.compile(optimizer='adam', loss=vae_loss , metrics=[reconstruction_loss, kl_loss])
vae.fit(x_data,x_data, epochs=50, batch_size=64)
def plot_images(rows, cols, images, title):

    grid = np.zeros(shape=(rows*64, cols*64, 3))

    for row in range(rows):

        for col in range(cols):

            grid[row*64:(row+1)*64, col*64:(col+1)*64, :] = images[row*cols + col]



    plt.figure(figsize=(20,20))       

    plt.imshow(grid)

    plt.title(title)

    plt.show()
# ORIGINAL IMAGES



predictions = x_data[:100]

plot_images(10,10,predictions,"ORIGINAL FACES")
# RECONSTRUCTION OF ORIGINAL IMAGES



predictions  = vae.predict(x_data[:100])

plot_images(10,10,predictions, "RECONSTRUCTED FACES")
#NEW FACES GENERATED FROM RANDOM NOISE



predictions= decoder.predict(np.random.randn(100, latent_dims))

plot_images(10,10,predictions, "RANDOM IMAGINED FACES")