from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image

from keras.preprocessing.image import load_img

from keras.applications.resnet50 import preprocess_input, decode_predictions

from keras.models import Model

import numpy as np

from os import listdir, walk

from os.path import isfile, join

import itertools

import sys,requests

from matplotlib import pyplot as plt

import pandas as pd
def getAllFilesInDirectory(directoryPath: str):

    ''' Helper function to read all the files in the directory by Kaggle '''

    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]



def load_image(img_path, show=False):

    ''' Helper function to load and show images using Keras'''

    img = image.load_img(img_path, target_size=(224, 224))

    img_tensor = image.img_to_array(img)

    img_tensor = np.expand_dims(img_tensor, axis=0)

    if show:

        plt.imshow(img_tensor[0]/255)                           

        plt.axis('off')

        plt.show()

# Missing yamper & boltund



canid_pokemon_imgs = ['vulpix.png','ninetales.jpg','growlithe.png','arcanine.png',

                        'eevee.png','vaporeon.png','jolteon.png','flareon.png',

                        'espeon.png','umbreon.png','snubbull.png','granbull.png',

                        'houndour.png','houndoom.png','raikou.png','entei.png',

                        'suicune.png','poochyena.png','mightyena.png','electrike.png',

                        'manectric.png','absol.png','riolu.png','lucario.png',

                        'leafeon.png','glaceon.png','lillipup.png','herdier.png',

                        'stoutland.png','zorua.png','zoroark.png','fennekin.png',

                        'braixen.png','delphox.png','furfrou.png','sylveon.png',

                        'rockruff.jpg','lycanroc-midday.jpg','zeraora.jpg']
# Pull all the canine-like images from the Pokemon dataset

pokemon_dir = '../input/pokemon-images-and-types/images/images'

all_pokemon_files = getAllFilesInDirectory(pokemon_dir)



canine_pokemon_files = []

for canine_file in canid_pokemon_imgs:

    if join(pokemon_dir, canine_file) in all_pokemon_files:

        canine_pokemon_files.append(join(pokemon_dir, canine_file))
# Check files to see if it looks about right

for each in canine_pokemon_files:

    print(each)
# What do these Pokemon look like?

for path in canine_pokemon_files:

    load_image(path, show = True)
def calcSimilarity(self_vect, feature_vectors):

    ''' Calculate Euclidian distances'''

    similar: dict = {}

    keys = [k for k,v in feature_vectors.items()]

    min_dist = 10000000

    for k,v in feature_vectors.items():

       dist=np.linalg.norm(self_vect-v)

       if(dist < min_dist):

           min_dist = dist

           similar = k

    return similar 
def predict(img_path : str, model: Model):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    return model.predict(x)
# def driver(poke_img):

    

#     # Predict with model

#     feature_vectors: dict = {}

#     model = ResNet50(weights='imagenet')

#     for img_path in getAllFilesInDirectory("../input/dog-breed-identification/train"):

#         feature_vectors[img_path] = predict(img_path,model)[0]

#     poke_vect = predict(poke_img,model)[0]

#     # print ("Computing image similarity")

#     result=calcSimilarity(poke_vect, feature_vectors)

#     print ("Your picture is most similar to : ",result)

# #     print(self_vect)

#     return result





def driver(self_img):

    feature_vectors: dict = {}

    model = ResNet50(weights='imagenet')

    print ("Reading images")

    for img_path in getAllFilesInDirectory("../input/dog-breed-identification/train"):

        feature_vectors[img_path] = predict(img_path,model)[0]

    self_vect = predict(self_img,model)[0]

    print ("Computing image similarity")

    result=calcSimilarity(self_vect, feature_vectors)

    print ("Your picture is most similar to : ",result)

#     print(self_vect)

    return result
feature_vectors: dict = {}

model = ResNet50(weights='imagenet')

for img_path in getAllFilesInDirectory("../input/dog-breed-identification/train"):

    feature_vectors[img_path] = predict(img_path,model)[0]
pokemon_vect = predict(canine_pokemon_files[0],model)[0]

result = driver(canine_pokemon_files[0])

load_image(canine_pokemon_files[0], show = True)

load_image(result, show = True)
# for each_poke in canine_pokemon_files:

#     print(each_poke)

#     result = driver(each_poke)

#     load_image(each_poke, show = True)

#     load_image(result, show = True)