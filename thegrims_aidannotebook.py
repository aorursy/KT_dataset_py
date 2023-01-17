class conf:

    sampling_rate = 44100

    duration = 2 # sec

    hop_length = 347*duration # to make time steps 128

    fmin = 20

    fmax = sampling_rate // 2

    n_mels = 128

    n_fft = n_mels * 20

    padmode = 'constant'

    samples = sampling_rate * duration
# rate = 44100





# import librosa



# print(data[0].shape)

# plt.imshow(data[0], interpolation='nearest')

# plt.show()



# t = data[2930].copy()

# t = np.compress([False, True], t, axis=2)

# print(t.shape)

# t = np.squeeze(t, axis=2)

# print(t.shape)

# plt.imshow(t, interpolation='nearest')

# plt.show()



# t = t.astype(float)

# print(type(t[0][0]))



# spectrogram = librosa.db_to_power(t)

# backconvert = librosa.feature.inverse.mel_to_audio(spectrogram,

#     sr=conf.sampling_rate,

#     n_fft=conf.n_fft,

#     hop_length=conf.hop_length,

#     pad_mode=conf.padmode,

#     fmin=conf.fmin,

#     fmax=conf.fmax)

# print('converted back')

# IPython.display.display(IPython.display.Audio(backconvert, rate=rate))



# res = librosa.feature.inverse.mel_to_audio(t)



# # c = np.zeros((826 , 954))

# # t = np.resize(t,(954, 954))

# # t = np.concatenate((t, c), axis=0)

# # t.shape = (954, 954)



# # t = t[0:128,0:128]

# t = np.transpose(t)



# print(t.shape)

# plt.imshow(t, interpolation='nearest')

# plt.show()

# mel_inverted_spectrogram = mel_to_spectrogram(t, mel_inversion_filter,

#                                                 spec_thresh=spec_thresh,

#                                                 shorten_factor=shorten_factor)

# print(mel_inverted_spectrogram.shape)

# fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(20,4))

# cax = ax.matshow(np.float32(mel_inverted_spectrogram), cmap=plt.cm.afmhot, origin='lower', aspect='auto',interpolation='nearest')

# fig.colorbar(cax)

# plt.title('Inverted mel Spectrogram')



# inverted_mel_audio = invert_pretty_spectrogram(np.transpose(t), fft_size = fft_size,

#                                             step_size = step_size, log = True, n_iter = 10)

# IPython.display.Audio(data=res, rate=rate)
from scipy.io import wavfile

from zipfile import ZipFile 

import librosa

import fnmatch

import os

from matplotlib import pyplot as plt



# iPython specific stuff

%matplotlib inline

import IPython.display

from ipywidgets import interact, interactive, fixed



# Packages we're using

import numpy as np

import copy

import scipy.ndimage



arr = os.listdir('/kaggle/working/')

if len(arr) < 5:

    with ZipFile('/kaggle/input/freesound-audio-tagging-2019/train_curated.zip', 'r') as zip:

        zip.extractall()
import csv

soundsDict = {}

with open('/kaggle/input/freesound-audio-tagging-2019/train_curated.csv', mode='r') as infile:

    reader = csv.reader(infile)

    for rows in reader:

        if rows[1] in soundsDict:

            soundsDict[rows[1]].append(rows[0])

        else:

            soundsDict[rows[1]] = [rows[0]]



rock = []

for i in soundsDict.keys():

    if 'guitar' in i:

        rock.append(i)

        print(i, len(soundsDict[i]))



# print('bass guitar: ', soundsDict['Bass_guitar'])
# 35688e71.wav

from skimage.transform import rescale, resize, downscale_local_mean

Acoustic = []

Electric = []

# 'Acoustic_guitar,Strum', 

for label in ['Bass_guitar']:

     for wav in soundsDict[label]:

            

#           Loading wav file to display and play

#             rate, playable = wavfile.read('/kaggle/working/'+wav)

#             print('label: ', label, ' wav ', wav)

#             IPython.display.display(IPython.display.Audio(playable, rate=rate))

            

#           Librosa load audio floating point data

            audio, sr = librosa.load('/kaggle/working/'+wav)

            print(np.shape(audio))

        

#             D = np.abs(librosa.stft(audio))**2 sr=sr, S=D)

            spectrogram = librosa.feature.melspectrogram(y=audio, 

                sr=conf.sampling_rate,

                n_mels=conf.n_mels,

                hop_length=conf.hop_length,

                n_fft=conf.n_fft,

                fmin=conf.fmin,

                fmax=conf.fmax)

    

#           Crops the spectrogram to be around 2 seconds in length and square in size

#             print(np.shape(spectrogram))

            spectrogram = spectrogram[ :, 0:128]

            if np.shape(spectrogram) == (128,128):

                spectrogram = librosa.power_to_db(spectrogram)

#                 spectrogram = np.absolute(spectrogram)

                plt.title(label)

                plt.imshow(spectrogram, interpolation='nearest')

                plt.show()

#               downscale spec to 32*32

                spectrogram = rescale(spectrogram, 0.25, anti_aliasing=False)

                plt.imshow(spectrogram, interpolation='nearest')

                plt.show()

            

#                 image_rescaled = rescale(spectrogram, 4, anti_aliasing=False)

#                 plt.imshow(image_rescaled, interpolation='nearest')

#                 plt.show()

                

                if label == 'Acoustic_guitar,Strum':

#                     print(spectrogram[0][0])

                    Acoustic.append(spectrogram)

                else:

                    Electric.append(spectrogram)



#                 spectrogram = librosa.db_to_power(spectrogram)

            else:

                print('trash')

            

            

#             backconvert = librosa.feature.inverse.mel_to_audio(spectrogram,

#                 sr=conf.sampling_rate,

#                 n_fft=conf.n_fft,

#                 hop_length=conf.hop_length,

#                 pad_mode=conf.padmode,

#                 fmin=conf.fmin,

#                 fmax=conf.fmax)

#             print('converted back')

#             IPython.display.display(IPython.display.Audio(backconvert, rate=rate))

                                                                          

        

            

#     for c, i in enumerate(os.listdir("/kaggle/working")):

#         if fnmatch.fnmatch(i, '*.wav'):

#             rate, real = wavfile.read('/kaggle/working/'+i)

#             print(i)

#             IPython.display.display(IPython.display.Audio(real, rate=rate))

#             audio, sr = librosa.load('/kaggle/working/'+i)

#         #     D = np.abs(librosa.stft(audio))**2 sr=sr, S=D)

#             spectrogram = librosa.feature.melspectrogram(y=audio, 

#                 sr=conf.sampling_rate,

#                 n_mels=conf.n_mels,

#                 hop_length=conf.hop_length,

#                 n_fft=conf.n_fft,

#                 fmin=conf.fmin,

#                 fmax=conf.fmax)



#             print(np.shape(spectrogram))

#             spectrogram = librosa.power_to_db(spectrogram)

#             spectrogram = spectrogram.astype(np.float32)



#             plt.imshow(spectrogram, interpolation='nearest')

#             plt.show()



#             spectrogram = librosa.db_to_power(spectrogram)



#             plt.imshow(spectrogram, interpolation='nearest')

#             plt.show()



#             backconvert = librosa.feature.inverse.mel_to_audio(spectrogram,

#                 sr=conf.sampling_rate,

#                 n_fft=conf.n_fft,

#                 hop_length=conf.hop_length,

#                 pad_mode=conf.padmode,

#                 fmin=conf.fmin,

#                 fmax=conf.fmax)

#             print('converted back')

#             IPython.display.display(IPython.display.Audio(backconvert, rate=rate))



#             if c == 2:

#                 break





# rate, real = wavfile.read('/kaggle/input/freesound-audio-tagging-2019/train_noisy/35688e71.wav')

# IPython.display.Audio(data=real, rate=rate)
Acoustic=np.asarray(Acoustic)

Electric=np.asarray(Electric)

Acoustic = np.expand_dims(Acoustic, axis=3)

Acoustic = np.expand_dims(Acoustic, axis=3)

print(np.shape(Acoustic))



# print(np.shape(Electric))
import keras

from keras import layers



# latent_dim = 10

# height = 128

# width = 128

# channels = 1



# generator_input = keras.Input(shape=(latent_dim,))



# # First, transform the input into a 16x16 128-channels feature map

# x = layers.Dense(40 * 64 * 64)(generator_input)

# x = layers.LeakyReLU()(x)

# x = layers.Reshape((64, 64, 40))(x)



# # Then, add a convolution layer

# x = layers.Conv2D(4096, 5, padding='same')(x)

# x = layers.LeakyReLU()(x)



# # Upsample to 32x32

# x = layers.Conv2DTranspose(4096, 4, strides=2, padding='same')(x)

# x = layers.LeakyReLU()(x)



# # Few more conv layers

# x = layers.Conv2D(4096, 5, padding='same')(x)

# x = layers.LeakyReLU()(x)

# x = layers.Conv2D(4096, 5, padding='same')(x)

# x = layers.LeakyReLU()(x)



latent_dim = 32

height = 32

width = 32

channels = 1



generator_input = keras.Input(shape=(latent_dim,))



# First, transform the input into a 16x16 128-channels feature map

x = layers.Dense(128 * 16 * 16)(generator_input)

x = layers.LeakyReLU()(x)

x = layers.Reshape((16, 16, 128))(x)



# Then, add a convolution layer

x = layers.Conv2D(256, 5, padding='same')(x)

x = layers.LeakyReLU()(x)



# Upsample to 32x32

x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)

x = layers.LeakyReLU()(x)



# Few more conv layers

x = layers.Conv2D(256, 5, padding='same')(x)

x = layers.LeakyReLU()(x)

x = layers.Conv2D(256, 5, padding='same')(x)

x = layers.LeakyReLU()(x)



# Produce a 32x32 1-channel feature map

x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)

generator = keras.models.Model(generator_input, x)

generator.summary()
discriminator_input = layers.Input(shape=(height, width, channels))

x = layers.Conv2D(128, 3)(discriminator_input)

x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.LeakyReLU()(x)

x = layers.Conv2D(128, 4, strides=2)(x)

x = layers.LeakyReLU()(x)

x = layers.Flatten()(x)

# discriminator_input = layers.Input(shape=(height, width, channels))

# x = layers.Conv2D(4096, 1)(discriminator_input)

# x = layers.LeakyReLU()(x)

# x = layers.Conv2D(4096, 4, strides=2)(x)

# x = layers.LeakyReLU()(x)

# x = layers.Conv2D(4096, 4, strides=2)(x)

# x = layers.LeakyReLU()(x)

# x = layers.Conv2D(4096, 4, strides=2)(x)

# x = layers.LeakyReLU()(x)

# x = layers.Flatten()(x)



# One dropout layer - important trick!

x = layers.Dropout(0.8)(x)



# Classification layer

x = layers.Dense(1, activation='sigmoid')(x)



discriminator = keras.models.Model(discriminator_input, x)

discriminator.summary()



# To stabilize training, we use learning rate decay

# and gradient clipping (by value) in the optimizer.

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0001, clipvalue=1.0, decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
# Set discriminator weights to non-trainable

# (will only apply to the `gan` model)

discriminator.trainable = False



gan_input = keras.Input(shape=(latent_dim,))

gan_output = discriminator(generator(gan_input))

gan = keras.models.Model(gan_input, gan_output)



gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)

gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
from keras.preprocessing import image



# Load CIFAR10 data

# (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()



# Select frog images (class 6) x_train[y_train.flatten() == 6]

x_train = Electric

print(np.shape(x_train))

# Normalize data

x_train =x_train.reshape(

    (x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

print(np.shape(x_train))



iterations = 2000

batch_size = 20

save_dir = '/kaggle/working/'



# Start training loop

start = 0

for step in range(iterations):

    # Sample random points in the latent space

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))



    # Decode them to fake images

    generated_images = generator.predict(random_latent_vectors)



    # Combine them with real images

    stop = start + batch_size

    real_images = x_train[start: stop]

    combined_images = np.concatenate([generated_images, real_images])



    # Assemble labels discriminating real from fake images

    labels = np.concatenate([np.ones((batch_size, 1)),

                             np.zeros((batch_size, 1))])

    # Add random noise to the labels - important trick!

    labels += 0.05 * np.random.random(labels.shape)



    # Train the discriminator

    d_loss = discriminator.train_on_batch(combined_images, labels)



    # sample random points in the latent space

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))



    # Assemble labels that say "all real images"

    misleading_targets = np.zeros((batch_size, 1))



    # Train the generator (via the gan model,

    # where the discriminator weights are frozen)

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    

    start += batch_size

    if start > len(x_train) - batch_size:

      start = 0



    # Occasionally save / plot

    if step % 100 == 0:

        # Save model weights

        gan.save_weights('gan.h5')



        # Print metrics

        print('discriminator loss at step %s: %s' % (step, d_loss))

        print('adversarial loss at step %s: %s' % (step, a_loss))



        # Save one generated image

        img = image.array_to_img(generated_images[0] * 255., scale=False)

        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))



        # Save one real image, for comparison

        img = image.array_to_img(real_images[0] * 255., scale=False)

        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))
import matplotlib.pyplot as plt



# Sample random points in the latent space

random_latent_vectors = np.random.normal(size=(10, latent_dim))



# Decode them to fake images

generated_images = generator.predict(random_latent_vectors)



for i in range(generated_images.shape[0]):



    test = np.squeeze(generated_images[i])

    plt.imshow(test, interpolation='nearest')

    plt.show()

    

    image_rescaled = rescale(test, 4, anti_aliasing=False)

    print(np.shape(image_rescaled))

    plt.imshow(image_rescaled, interpolation='nearest')

    plt.show()

    

    backconvert = librosa.feature.inverse.mel_to_audio(image_rescaled,

        sr=conf.sampling_rate,

        n_fft=conf.n_fft,

        hop_length=conf.hop_length,

        pad_mode=conf.padmode,

        fmin=conf.fmin,

        fmax=conf.fmax)

    print('converted back')

    IPython.display.display(IPython.display.Audio(backconvert, rate=conf.sampling_rate))

plt.show()
os.system('rm -rf /kaggle/working')