# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import libraries



import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import librosa as lb

from librosa import feature, power_to_db

import os

import glob

import pathlib

import datetime

import io

import os.path

from keras.models import Model

from keras import layers

import sys

from shutil import copyfile

import pickle



os.listdir('../input/audio-data/libs')



copyfile(src = "../input/audio-data/libs/libs/speech_vgg.py", dst = "../working/speech_vgg.py")

from speech_vgg import speechVGG
def build_dir(path):

    try:

        os.mkdir(path)

    except OSError:

        print ("Creation of the directory {} failed" .format(path))

    else:

        print ("Successfully created the directory {} " .format(path))



PATH = '../input/audio-data/audios_data/audios_data'



data_dir = pathlib.Path(PATH)



train_dir = pathlib.Path(os.path.join(PATH, 'training'))



val_dir = pathlib.Path(os.path.join(PATH, 'validation'))





sample_count = len(list(data_dir.glob('*/*/*.wav')))



class_names = np.array(sorted([item.name for item in train_dir.glob('*')]))

n_classes = int(len(class_names))



print("{} samples appartenant à {} classes : {}".format(sample_count, n_classes, class_names))

print("__________________________________________________________________________________________")



train_image_count = len(list(train_dir.glob('*/*.wav')))

val_image_count = len(list(val_dir.glob('*/*.wav')))



print("Répertoire d'entrainement : \n {}" .format(train_dir))

print("{} audios" .format(train_image_count))

print("__________________________________________________________________________________________")

print("Répertoire de validation : \n {}" .format(val_dir))

print("{} audios" .format(val_image_count))
class DataGenerator(tf.keras.utils.Sequence):

    "Generate data on the fly"

    def __init__(self, file_list, labels, batch_size=32, dim=(256,251), n_channels=1, n_classes=None, shuffle=True):

        "Initialisation"

        self.dim = dim

        self.batch_size = batch_size

        self.labels = labels

        self.file_list = file_list

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.shuffle = shuffle

        self.on_epoch_end()

        

    def __len__(self):

        "Nombre de batch par époque"

        return int(np.floor(len(self.file_list) / self.batch_size))

    

    def __getitem__(self, index):

        "Création d'un batch"

        # Génération de l'indice du batch

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        

        "Liste des fichiers"

        file_list_tmp = [self.file_list[k] for k in indexes]

        

        "Génération du batch"

        X, y = self.__data_generation(file_list_tmp)

        return X, y

    def on_epoch_end(self):

        "Mise à jour des indices après chaque époque"

        self.indexes = np.arange(len(self.file_list))

        if self.shuffle == True:

            np.random.shuffle(self.indexes)



    def __data_generation(self, file_list_tmp):

        "Génération des données d'un batch"

        # Initialisation

        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        y = np.empty((self.batch_size), dtype=int)

        

        # Génération des données

        for i, ID in enumerate(file_list_tmp):

            # Sample

            spec = spectrogram_DL4S_utils(ID)

            X[i,] = np.expand_dims(spec, axis=2)

            

            # Classe

            y[i] = self.labels[ID]

    

        return X, tf.keras.utils.to_categorical(y, num_classes = self.n_classes)
# Définition de la taille des spectrogrammes

height = 128

width = 251

channels = 1



# Définition des paramètres du data generator

params = {'dim': (height, width),

          'batch_size': 64,

          'n_classes': n_classes,

          'n_channels': channels,

          'shuffle': True

         }





# Création des dictionnaires du dataset

# Noms des fichiers pour les dossier 'training' et 'validation'



ext = '.wav'

sample_dic = {}

label_dic = {}
for key in ['training', 'validation']:

    file_list = glob.glob(str(data_dir) + "/" + key + "/*/*")

    

    #Extension des fichiers

    sample_dic[key] = file_list

    

    #Labels associés à chaque fichier

    for file in file_list:

        label = tf.strings.split(file, os.path.sep)[-2]



        # The second to last is the class-directory

        label_dic[file] = np.argmax(label == class_names)
train_gen = DataGenerator(sample_dic['training'], label_dic, **params)

val_gen = DataGenerator(sample_dic['validation'], label_dic, **params)
# Charger le modèle téléchargé (h5 file)

model_path = "../input/audio-data/model_100h.h5"

print (model_path)



'''

##Appel du modèle SpeechVGG

model = speechVGG(

            include_top=True,

            input_shape=(128, 251, 1),

            classes=n_classes,

            pooling=None,

            weights=None,

            transfer_learning=True

        )



model.summary()



'''



model = tf.keras.models.Sequential([

#    tf.keras.layers.Lambda(spectrogram_DL4S_utils, output_shape=(,))

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(128, 251,1)),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(len(class_names), activation='softmax')

])



model.summary()

def spectrogram_DL4S_utils(audio, fft_win=512, hop=128, nmels=128):



    '''

    audio = AudioWav(ID)

    spectro = audio.spectrogram(fft_win = fft_win, hop = hop, nmels=nmels)

    spectro_db_normalized = spectro/-80.

    '''



    y, sr = lb.load(audio, sr=8000)

    spectro = lb.feature.melspectrogram(y=y, sr=sr, n_fft = 512, hop_length = 128, n_mels= 128)

    spectro_db_normalized = spectro/-80.

    #print(spectro_db_normalized.shape)

    return spectro_db_normalized
# Compile model

model.compile(

            optimizer=tf.keras.optimizers.Adam(),

            loss='categorical_crossentropy',

            metrics=['acc']

            )

                                                                                

#Entrainer le model



history = model.fit(train_gen, epochs=5, validation_data= val_gen)
# Save the entire model as a SavedModel.

!mkdir saved_model

model.save('../working/saved_model/CNN.h5')