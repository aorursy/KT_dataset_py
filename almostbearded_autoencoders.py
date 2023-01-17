import matplotlib.pyplot as plt

import random

import tensorflow as tf

from tensorflow import keras

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D

from keras.models import Model, load_model

from keras import regularizers

from keras.callbacks import EarlyStopping

from scipy.io import loadmat

import json

import numpy as np

import time
def plot_loss(history):

    plt.plot(history['loss'])

    plt.plot(history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()



def plot_image(image):

    plt.imshow(image.reshape((28, 28)))    

    return



def plot_images_sample(images):

    plt.figure(figsize=(8, 5))

    plt.subplots_adjust(wspace=0, hspace=0)

    random_indices = random.sample(range(len(images)), 40)

    for i in range(40):

        plt.gray()

        subplot = plt.subplot(5, 8, i + 1)

        subplot.get_xaxis().set_visible(False)

        subplot.get_yaxis().set_visible(False)



        plot_image(images[random_indices[i]])



def plot_comparison(input_images, output_images):

    plt.figure(figsize=(18, 4))

    plt.subplots_adjust(wspace=0, hspace=0)

    random_indices = random.sample(range(len(input_images)), 12)

    for i in range(12):

        plt.gray()

        subplot = plt.subplot(2, 12, i + 1)

        subplot.get_xaxis().set_visible(False)

        subplot.get_yaxis().set_visible(False)

        plot_image(input_images[random_indices[i]])

        subplot = plt.subplot(2, 12, i + 1 + 12)

        subplot.get_xaxis().set_visible(False)

        subplot.get_yaxis().set_visible(False)

        plot_image(output_images[random_indices[i]])



def save_to_file(var, filename):

    js = json.dumps(var)

    f = open(filename, "w")

    f.write(js)

    f.close()



def load_from_file(filename):

    f = open(filename, "r")

    js = f.read()

    return json.loads(js)
training_epochs = 100

train_to_test_ratio = 0.85

input_size = 28 * 28

encoded_size = 7 * 7

activation = 'elu'

optimizer = 'adadelta'

loss = 'binary_crossentropy'

callbacks = [EarlyStopping(monitor='val_loss', patience=3)]

use_serialized = False
# load the dataset

mnist = loadmat("../input/mnist-original/mnist-original")



# get the image data from the dataset

images = mnist["data"].T



# convert image data into float format

images = images / 255



# split into training and test images

train_count = round(len(images) * train_to_test_ratio)

test_count = len(images) - train_count

train_images = images[:train_count]

test_images = images[-test_count:]



print("all images:", images.shape)

print("train images:", train_images.shape)

print("test images:", test_images.shape)
plot_images_sample(train_images)
if (use_serialized):

    vanilla_autoencoder = load_model('vanilla.h5')

    vanilla_encoder = load_model('vanilla_encoder.h5')

    vanilla_decoder = load_model('vanilla_decoder.h5')



else:

    input_layer = Input(shape=(input_size,))

    

    # encoder

    encoded_layer = Dense(encoded_size, activation=activation)(input_layer)



    # decoder

    decoded_layer = Dense(input_size, activation='sigmoid')(encoded_layer)



    # autoencoder

    vanilla_autoencoder = Model(input_layer, decoded_layer)

    vanilla_autoencoder.name = "vanilla autoencoder"



    # separate encoder and decoder models

    vanilla_encoder = Model(input_layer, encoded_layer)

    vanilla_encoder.name = "vanilla encoder"



    encoded_input_layer = Input(shape=(encoded_size,))

    vanilla_decoder = Model(encoded_input_layer, vanilla_autoencoder.layers[-1](encoded_input_layer))

    vanilla_decoder.name = "vanilla decoder"



# print model summaries

vanilla_autoencoder.summary()

vanilla_encoder.summary()

vanilla_decoder.summary()
if (use_serialized):

    history = load_from_file('vanilla_history.json')

    

else:

    vanilla_autoencoder.compile(optimizer=optimizer, loss=loss)

    start = time.time()

    history = vanilla_autoencoder.fit(train_images, train_images, 

                            epochs=training_epochs, 

                            batch_size=256, 

                            shuffle=True, 

                            validation_data=(test_images, test_images),

                            callbacks=callbacks).history

    print('training time:', time.time() - start)

    save_to_file(history, 'vanilla_history.json')

    vanilla_autoencoder.save('vanilla.h5')

    vanilla_encoder.save('vanilla_encoder.h5')

    vanilla_decoder.save('vanilla_decoder.h5')    



plot_loss(history)

print("loss:", history["loss"][-1])

print("val_loss:", history["val_loss"][-1])

# encode images

encoded_images = vanilla_encoder.predict(test_images)

print("encoded images:", encoded_images.shape)



#decode images

decoded_images = vanilla_decoder.predict(encoded_images)

print("decoded images:", decoded_images.shape)
plot_comparison(test_images, decoded_images)
if (use_serialized):

    multilayer_autoencoder = load_model('multilayer.h5')

    multilayer_encoder = load_model('multilayer_encoder.h5')

    multilayer_decoder = load_model('multilayer_decoder.h5')



else:

    input_layer = Input(shape=(input_size,))



    # encoder

    hidden_encoder_layer = Dense(round((input_size + encoded_size) / 4), activation=activation)(input_layer)

    encoded_layer = Dense(encoded_size, activation=activation)(hidden_encoder_layer)



    # decoder

    hidden_decoder_layer = Dense(round((input_size + encoded_size) / 4), activation=activation)(encoded_layer)

    decoded_layer = Dense(input_size, activation='sigmoid')(hidden_decoder_layer)



    # autoencoder

    multilayer_autoencoder = Model(input_layer, decoded_layer)

    multilayer_autoencoder.name = "multilayer autoencoder"



    # separate encoder and decoder models

    multilayer_encoder = Model(input_layer, encoded_layer)

    multilayer_encoder.name = "multilayer encoder"



    encoded_input_layer = Input(shape=(encoded_size,))

    multilayer_decoder = Model(encoded_input_layer, multilayer_autoencoder.layers[-1](multilayer_autoencoder.layers[-2](encoded_input_layer)))

    multilayer_decoder.name = "multilayer decoder"



# print model summaries

multilayer_autoencoder.summary()

multilayer_decoder.summary()

multilayer_encoder.summary()

if (use_serialized):

    history = load_from_file('multilayer_history.json')

    

else:

    multilayer_autoencoder.compile(optimizer=optimizer, loss=loss)

    start = time.time()

    history = multilayer_autoencoder.fit(train_images, train_images, 

                            epochs=training_epochs, 

                            batch_size=256, 

                            shuffle=True, 

                            validation_data=(test_images, test_images),

                            callbacks=callbacks).history

    print('training time:', time.time() - start)

    save_to_file(history, 'multilayer_history.json')

    multilayer_autoencoder.save('multilayer.h5')

    multilayer_encoder.save('multilayer_encoder.h5')        

    multilayer_decoder.save('multilayer_decoder.h5') 

    

plot_loss(history)   

print("loss:", history["loss"][-1])

print("val_loss:", history["val_loss"][-1])
# encode images

encoded_images = multilayer_encoder.predict(test_images)

print("encoded images:", encoded_images.shape)



#decode images

decoded_images = multilayer_decoder.predict(encoded_images)

print("decoded images:", decoded_images.shape)
plot_comparison(test_images, decoded_images)
if (use_serialized):

    convolutional_autoencoder = load_model('convolutional.h5')

    convolutional_encoder = load_model('convolutional_encoder.h5')

    convolutional_decoder = load_model('convolutional_decoder.h5')



else:

    input_layer = Input(shape=(28, 28, 1))



    ## encoder

    # first convolution (outputs 14x14)

    layer = Conv2D(16, (3, 3), activation=activation, padding='same')(input_layer)

    layer = MaxPooling2D((2, 2), padding='same')(layer)

    

    # second convolution (outputs 7x7)

    layer = Conv2D(16, (3, 3), activation=activation, padding='same')(layer)

    encoded_layer = MaxPooling2D((2, 2), padding='same')(layer)

    

    ## decoder

    # first deconvolution (outputs 14x14)

    layer = Conv2D(16, (3, 3), activation=activation, padding='same')(encoded_layer)

    layer = UpSampling2D((2, 2))(layer)

    

    # second deconvolution (outputs 28x28)

    layer = Conv2D(16, (3, 3), activation=activation, padding='same')(layer)

    layer = UpSampling2D((2, 2))(layer)



    # TODO: why is this layer needed?

    decoded_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)



    # autoencoder

    convolutional_autoencoder = Model(input_layer, decoded_layer)

    convolutional_autoencoder.name = "convolutional autoencoder"



    # separate encoder and decoder models

    convolutional_encoder = Model(input_layer, encoded_layer)

    convolutional_encoder.name = "convolutional encoder"



    encoded_input_layer = Input(shape=(7, 7, 16))

    convolutional_decoder = Model(encoded_input_layer,

        convolutional_autoencoder.layers[-1](

            convolutional_autoencoder.layers[-2](

                convolutional_autoencoder.layers[-3](

                    convolutional_autoencoder.layers[-4](

                        convolutional_autoencoder.layers[-5](

                            encoded_input_layer))))))

              

    convolutional_decoder.name = "convolutional decoder"



# print model summaries

convolutional_autoencoder.summary()

convolutional_encoder.summary()

convolutional_decoder.summary()

if (use_serialized):

    history = load_from_file('convolutional_history.json')

    

else:

    convolutional_autoencoder.compile(optimizer=optimizer, loss=loss)

    

    reshaped_train_images = train_images.reshape((len(train_images), 28, 28, 1))

    reshaped_test_images = test_images.reshape((len(test_images), 28, 28, 1))



    start = time.time()

    history = convolutional_autoencoder.fit(reshaped_train_images, reshaped_train_images, 

                            epochs=training_epochs, 

                            batch_size=256, 

                            shuffle=True, 

                            validation_data=(reshaped_test_images, reshaped_test_images),

                            callbacks=callbacks).history

    print('training time:', time.time() - start)

    save_to_file(history, 'convolutional_history.json')

    convolutional_autoencoder.save('convolutional.h5')

    convolutional_encoder.save('convolutional_encoder.h5')        

    convolutional_decoder.save('convolutional_decoder.h5')    

    

plot_loss(history)

print("loss:", history["loss"][-1])

print("val_loss:", history["val_loss"][-1])
# encode images

encoded_images = convolutional_encoder.predict(reshaped_test_images)

print("encoded images:", encoded_images.shape)



#decode images

decoded_images = convolutional_decoder.predict(encoded_images)

print("decoded images:", decoded_images.shape)
plot_comparison(test_images, decoded_images)
if (use_serialized):

    sparse_autoencoder = load_model('sparse.h5')

    sparse_encoder = load_model('sparse_encoder.h5')

    sparse_decoder = load_model('sparse_decoder.h5')



else:

    input_layer = Input(shape=(input_size,))

    

    # encoder

    encoded_layer = Dense(input_size, activation=activation,

        activity_regularizer=regularizers.l1(10e-7))(input_layer)



    # decoder

    decoded_layer = Dense(input_size, activation='sigmoid')(encoded_layer)



    # autoencoder

    sparse_autoencoder = Model(input_layer, decoded_layer)

    sparse_autoencoder.name = "sparse autoencoder"



    # separate encoder and decoder models

    sparse_encoder = Model(input_layer, encoded_layer)

    sparse_encoder.name = "sparse encoder"



    encoded_input_layer = Input(shape=(input_size,))

    sparse_decoder = Model(encoded_input_layer, 

       sparse_autoencoder.layers[-1](encoded_input_layer))

    sparse_decoder.name = "sparse decoder"



# print model summaries

sparse_autoencoder.summary()

sparse_encoder.summary()

sparse_decoder.summary()
if (use_serialized):

    history = load_from_file('sparse_history.json')

    

else:

    sparse_autoencoder.compile(optimizer=optimizer, loss=loss)

    start = time.time()

    history = sparse_autoencoder.fit(train_images, train_images, 

                            epochs=training_epochs, 

                            batch_size=256, 

                            shuffle=True, 

                            validation_data=(test_images, test_images),

                            callbacks=callbacks).history

    print('training time:', time.time() - start)

    save_to_file(history, 'sparse_history.json')

    sparse_autoencoder.save('sparse.h5')

    sparse_encoder.save('sparse_encoder.h5')

    sparse_decoder.save('sparse_decoder.h5')    



plot_loss(history)

print("loss:", history["loss"][-1])

print("val_loss:", history["val_loss"][-1])

# encode images

encoded_images = sparse_encoder.predict(test_images)

print("encoded images:", encoded_images.shape)



#decode images

decoded_images = sparse_decoder.predict(encoded_images)

print("decoded images:", decoded_images.shape)
plot_comparison(test_images, decoded_images)
noise_factor = 0.4

noisy_train_images = np.clip(train_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=train_images.shape), 0, 1) 

noisy_test_images = np.clip(test_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=test_images.shape), 0, 1)



plot_images_sample(noisy_train_images)
if (use_serialized):

    denoising_autoencoder = load_model('denoising.h5')

    denoising_encoder = load_model('denoising_encoder.h5')

    denoising_decoder = load_model('denoising_decoder.h5')



else:

    input_layer = Input(shape=(28, 28, 1))



    ## encoder

    # first convolution (outputs 14x14)

    layer = Conv2D(32, (3, 3), activation=activation, padding='same')(input_layer)

    layer = MaxPooling2D((2, 2), padding='same')(layer)

    

    # second convolution (outputs 7x7)

    layer = Conv2D(32, (3, 3), activation=activation, padding='same')(layer)

    encoded_layer = MaxPooling2D((2, 2), padding='same')(layer)

    

    ## decoder

    # first deconvolution (outputs 14x14)

    layer = Conv2D(32, (3, 3), activation=activation, padding='same')(encoded_layer)

    layer = UpSampling2D((2, 2))(layer)

    

    # second deconvolution (outputs 28x28)

    layer = Conv2D(32, (3, 3), activation=activation, padding='same')(layer)

    layer = UpSampling2D((2, 2))(layer)



    decoded_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(layer)



    # autoencoder

    denoising_autoencoder = Model(input_layer, decoded_layer)

    denoising_autoencoder.name = "denoising autoencoder"



    # separate encoder and decoder models

    denoising_encoder = Model(input_layer, encoded_layer)

    denoising_encoder.name = "denoising encoder"



    encoded_input_layer = Input(shape=(7, 7, 32))

    denoising_decoder = Model(encoded_input_layer,

        denoising_autoencoder.layers[-1](

            denoising_autoencoder.layers[-2](

                denoising_autoencoder.layers[-3](

                    denoising_autoencoder.layers[-4](

                        denoising_autoencoder.layers[-5](

                            encoded_input_layer))))))

              

    denoising_decoder.name = "denoising decoder"



# print model summaries

denoising_autoencoder.summary()

denoising_encoder.summary()

denoising_decoder.summary()
if (use_serialized):

    history = load_from_file('denoising_history.json')

    

else:

    denoising_autoencoder.compile(optimizer=optimizer, loss=loss)

   

    reshaped_noisy_train_images = noisy_train_images.reshape((len(noisy_train_images), 28, 28, 1))

    reshaped_train_images = train_images.reshape((len(train_images), 28, 28, 1))

    reshaped_test_images = test_images.reshape((len(test_images), 28, 28, 1))

    reshaped_noisy_test_images = noisy_test_images.reshape((len(noisy_test_images), 28, 28, 1))

    

    start = time.time()

    history = denoising_autoencoder.fit(reshaped_noisy_train_images, reshaped_train_images, 

                            epochs=training_epochs, 

                            batch_size=256, 

                            shuffle=True, 

                            validation_data=(reshaped_noisy_test_images, reshaped_test_images)).history

    print('training time:', time.time() - start)

    save_to_file(history, 'denoising_history.json')

    denoising_autoencoder.save('denoising.h5')

    denoising_encoder.save('denoising_encoder.h5')        

    denoising_decoder.save('denoising_decoder.h5')    

    

plot_loss(history)

print("loss:", history["loss"][-1])

print("val_loss:", history["val_loss"][-1])
# encode images

encoded_images = denoising_encoder.predict(reshaped_noisy_test_images)

print("encoded images:", encoded_images.shape)



#decode images

decoded_images = denoising_decoder.predict(encoded_images)

print("decoded images:", decoded_images.shape)
plot_comparison(noisy_test_images, decoded_images)