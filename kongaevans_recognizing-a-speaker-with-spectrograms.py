# Import the required packages

import os

import numpy as np 

import tensorflow as tf

import matplotlib.pyplot as plt

import pathlib

import librosa.display

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
# a package to compute speech features, implemented using tf.keras

try:

    from spela.spectrogram import Spectrogram 

    from spela.melspectrogram import Melspectrogram

except:

    !pip install spela

    from spela.spectrogram import Spectrogram 

    from spela.melspectrogram import Melspectrogram
# disable eager execution, my model couldn't train well while on eager mode

tf.compat.v1.disable_eager_execution()
# Get the data directories

data_dir = "../input/speaker-recognition-dataset/16000_pcm_speeches/"
# inspect the folders inside the dataset

os.listdir(data_dir)
# for now we are concerned with the four speakers

# lets get as a sample data from one of the speakers

nelson_madela = [item for item in os.listdir(data_dir + "Nelson_Mandela")]

nelson_madela[:10]
# lets create a function that takes in a raw wavfile and computes a spectrogram then plots it

def compute_spectrogram_melspectrogram_and_plot(wav_dir, compute_type):

    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:

        wav_filename_placeholder = tf.compat.v1.placeholder(tf.compat.v1.string, [])

        wav_loader = tf.io.read_file(wav_filename_placeholder)

        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)

        wav_data = sess.run(

        wav_decoder, feed_dict={

            wav_filename_placeholder: wav_dir

        }).audio.flatten()

        sess.close()

    # audio has a sample rate of 16000 and the produced wav has a shape of (16000, 1)

    # reshape to (1, 1600)

    wav = wav_data.reshape(1, 16000)

    wav_new = wav[np.newaxis, :] # introduce a new axis to have a shape of (1, 1, 16000)

    height = wav_new.shape[1]

    width = wav_new.shape[2]

    # create a model to compute spectrogram

    model = tf.keras.Sequential() 

    if compute_type == "spectrogram":

        model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(height, width),

                            return_decibel_spectrogram=True, power_spectrogram=2.0,

                            trainable_kernel=False, name='static_stft'))

    elif compute_type == "melspectrogram":

        model.add(Melspectrogram(sr=16000, n_mels=128,n_dft=512, n_hop=256,

                            input_shape=(height, width), return_decibel_melgram=True,

                            trainable_kernel=False, name='melgram'))

   

    # producing a spectrogram/melspectrogram from the model

    pred = model.predict(x=wav_new)



    if tf.keras.backend.image_data_format() == "channel_first":

        result = pred[0, 0]

    else:

        result = pred[0, :, :, 0]



    # show the spectrogram/melspectrogram

    librosa.display.specshow(result, y_axis='linear', sr=16000)
compute_spectrogram_melspectrogram_and_plot(data_dir + "Nelson_Mandela/" + nelson_madela[0], "spectrogram")
compute_spectrogram_melspectrogram_and_plot(data_dir + "Nelson_Mandela/" + nelson_madela[0], "melspectrogram")
# get wav paths

def get_wav_paths(speaker):

    speaker_path = data_dir + speaker

    all_paths = [item for item in os.listdir(speaker_path)]

    return all_paths
nelson_mandela_paths = get_wav_paths("Nelson_Mandela")

margaret_thatcher_paths = get_wav_paths("Magaret_Tarcher")

benjamin_netanyau_paths = get_wav_paths("Benjamin_Netanyau")

jens_stoltenberg_paths = get_wav_paths( 'Jens_Stoltenberg')

julia_gillard_paths = get_wav_paths("Julia_Gillard")
# load the data

def load_wav(wav_path, speaker):

    with tf.compat.v1.Session(graph=tf.compat.v1.Graph()) as sess:

        wav_path = data_dir +speaker + "/"+ wav_path

        wav_filename_placeholder = tf.compat.v1.placeholder(tf.compat.v1.string, [])

        wav_loader = tf.io.read_file(wav_filename_placeholder)

        wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)

        wav_data = sess.run(

            wav_decoder, feed_dict={

                wav_filename_placeholder: wav_path

            }).audio.flatten().reshape((1, 16000))

        sess.close()

    return wav_data

    
# create training data

def generate_training_data(speaker_paths, speaker, label):

    wavs, labels = [], []

    for i in tqdm(speaker_paths):

        wav = load_wav(i, speaker)

        wavs.append(wav)

        labels.append(label)

    return wavs, labels

nelson_mandela_wavs, nelson_mandela_labels = generate_training_data(nelson_mandela_paths, "Nelson_Mandela", 0) 

margaret_thatcher_wavs, margaret_thatcher_labels = generate_training_data(margaret_thatcher_paths, "Magaret_Tarcher", 1) 

benjamin_netanyau_wavs, benjamin_netanyau_labels = generate_training_data(benjamin_netanyau_paths, "Benjamin_Netanyau", 2) 

jens_stoltenberg_wavs, jens_stoltenberg_labels = generate_training_data(jens_stoltenberg_paths, "Jens_Stoltenberg", 3) 

julia_gillard_wavs, julia_gillard_labels = generate_training_data(julia_gillard_paths, "Julia_Gillard", 4) 
# remove the extra wav for Julia Gillard

julia_gillard_labels = julia_gillard_labels[1:]

julia_gillard_wavs = julia_gillard_wavs[1:]
all_wavs = nelson_mandela_wavs + margaret_thatcher_wavs + benjamin_netanyau_wavs + jens_stoltenberg_wavs + julia_gillard_wavs

all_labels = nelson_mandela_labels + margaret_thatcher_labels + benjamin_netanyau_labels + jens_stoltenberg_labels + julia_gillard_labels
# split the dataset into trainin and testing set\

train_wavs, test_wavs, train_labels, test_labels = train_test_split(all_wavs, all_labels, test_size=0.2)
train_x, train_y = np.array(train_wavs), np.array(train_labels)

test_x, test_y = np.array(test_wavs), np.array(test_labels)
train_y = tf.keras.utils.to_categorical(train_y)

test_y = tf.keras.utils.to_categorical(test_y)
# create a model

def create_model(speech_feature):

    model = tf.keras.Sequential()

    if speech_feature == "spectrogram":

        model.add(Spectrogram(n_dft=512, n_hop=256, input_shape=(1, 16000),

                            return_decibel_spectrogram=True, power_spectrogram=2.0,

                            trainable_kernel=False, name='static_stft'))

    elif speech_feature == "melspectrogram":

        model.add(Melspectrogram(sr=16000, n_mels=128,n_dft=512, n_hop=256,

                            input_shape=(1 , 16000),return_decibel_melgram=True,

                            trainable_kernel=False, name='melgram'))

   



    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))

    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))



    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(5, activation="softmax"))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4)

            , loss = "categorical_crossentropy"

            , metrics = ["accuracy"])

    return model
# spectrogam model

model = create_model("spectrogram")
model.summary()
model.fit(x=train_x, y=train_y, epochs=10, validation_data=(test_x, test_y))
# melspectrogram model

model = create_model("melspectrogram")
model.summary()
model.fit(x=train_x, y=train_y, epochs=10, validation_data=(test_x, test_y))