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
def getAllFilesInDirectory(directoryPath: str):

    return [(directoryPath + "/" + f) for f in listdir(directoryPath) if isfile(join(directoryPath, f))]
def predict(img_path : str, model: Model):

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    return model.predict(x)
def calcSimilarity(self_vect, feature_vectors):

    similar: dict = {}

    keys = [k for k,v in feature_vectors.items()]

    min_dist = 10000000

    for k,v in feature_vectors.items():

       dist=np.linalg.norm(self_vect-v)

       if(dist < min_dist):

           min_dist = dist

           similar = k

    return similar 
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))

    img_tensor = image.img_to_array(img)

    img_tensor = np.expand_dims(img_tensor, axis=0)

    if show:

        plt.imshow(img_tensor[0]/255)                           

        plt.axis('off')

        plt.show()
def driver(self_img):

    feature_vectors: dict = {}

    model = ResNet50(weights='imagenet')

    print ("Reading images")

    for img_path in getAllFilesInDirectory("../input/pokemon-images-and-types/images/images"):

        feature_vectors[img_path] = predict(img_path,model)[0]

    self_vect = predict(self_img,model)[0]

    print ("Computing image similarity")

    result=calcSimilarity(self_vect, feature_vectors)

    print ("Your picture is most similar to : ",result)

#     print(self_vect)

    return result
f = open('Self.jpg','wb')

r = requests.get("https://scontent.fblr1-3.fna.fbcdn.net/v/t1.0-9/67348703_10219758215926660_18338638475558912_n.jpg?_nc_cat=107&_nc_oc=AQmBneQL7Bgrr19jONnTfV8y9Er05NR1PdUQ7xb9723SOS1xZeQcJ7OBKYPSVp3_gZs&_nc_ht=scontent.fblr1-3.fna&oh=8d06ca925e759113191c0d05d81469e8&oe=5DA7A134")

f.write(r.content)

f.close()
self_img_path = "../working/Self.jpg"

result = driver(self_img_path)
load_image(self_img_path, show = True)

load_image(result, show = True)