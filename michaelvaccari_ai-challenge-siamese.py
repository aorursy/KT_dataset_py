import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers, Model

from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda

import numpy as np

import json

import requests

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm

import matplotlib.patches as patches

from PIL import Image
# import delle immagini di prova e ridimensionamento

image_shape = (64, 64, 3)

path = '/kaggle/input/provaimmagini/'

test_imgs = np.zeros((1,128,128,3), dtype=np.uint8)

test_imgs1 = np.zeros((1,128,128,3), dtype=np.uint8)

a = Image.open(path+'immagine1.jpg','r')

b = Image.open(path+'immagine2.jpg', 'r')

a.resize((64, 64))

a.resize((64, 64))

test_imgs[0] = np.asarray(a)

test_imgs1[0] = np.asarray(b)

a.close()

b.close()
plt.imshow(test_imgs[0].astype(np.uint8), extent=[0, 128, 0, 128])
plt.imshow(test_imgs1[0].astype(np.uint8), extent=[0, 128, 0, 128])
# definizione di un modello semplificato per mostrarne il funzionamento

def build_model(id):

    input = keras.Input(image_shape)

    rn = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input, pooling=None)

    rn.layers.pop()

    rn.outputs = [rn.layers[-1].output]

    for layer in rn.layers:

        layer.trainable = False

        layer._name = layer.name + str(id)

    return Model(inputs=input, outputs=rn.outputs)

    
# creazione delle 2 ResNet

left, right = build_model(1), build_model(2)
# quando nella ai inseriamo 2 immagini non identiche la differenza dei 2 tensori in uscita dalle cnn aumenta;

# rimane comunque bassa, essendo le immagini simili (da notare che il valore non va da 0 a 1)

result_left, result_right = left.predict(test_imgs), right.predict(test_imgs1)

distance_function = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))

distance = distance_function([result_left, result_right])



np.mean(distance)
# se invece inserimao 2 immagini identiche la differenza tende a zero essendo le caratteristiche estratte uguali

result_left, result_right = left.predict(test_imgs), right.predict(test_imgs)

distance_function = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))

distance = distance_function([result_left, result_right])



np.mean(distance)
# questo è il modello completo che però mostra risultati pessimi per la mancanza di allenamento

def build_full_model():

    input1 = keras.Input(image_shape)

    rn1 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input1, pooling=None)

    rn1.layers.pop()

    rn1.outputs = [rn1.layers[-1].output]

    for layer in rn1.layers[:-3]:

        layer.trainable = False

    for layer in rn1.layers:

        layer._name = layer.name+'1'



    x = rn1.output

    x = Dense(1024, activation='relu')(x)

    #x = Dropout(0.1)(x)

    x = Dense(512, activation='relu')(x)

    out1 = Flatten()(x)





    input2 = keras.Input(image_shape)

    rn2 = keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=input2, pooling=None)

    rn2.layers.pop()

    rn2.outputs = [rn2.layers[-1].output]

    for layer in rn2.layers[:-3]:

        layer.trainable = False

    for layer in rn2.layers:

        layer._name = layer.name+'2'



    y = rn2.output

    y = Dense(1024, activation='relu')(y)

    y = Dense(512, activation='relu')(y)

    out2 = Flatten()(y)



    distance_function = Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))

    distance = distance_function([out1, out2])

    

    #sigmoid = Dense(1, activation='sigmoid')(distance)

    # non siamo riusciti ad implementare il layer di output con funzione di attivazione sigmoid per un errore dimensionale

    # ma è necessario per il training



    return Model(inputs=[input1, input2], outputs=distance)

# creazione del modello completo di siamese

full_model = build_full_model()
# si può vedere come i risultati sono sbagliati per l'aggiunta di layer senza allenamento

result = full_model.predict([test_imgs, test_imgs])

np.mean(result)