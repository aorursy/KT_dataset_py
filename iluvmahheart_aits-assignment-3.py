import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt

import cv2



print(os.listdir("../input"))
DATADIR = "../input/fruits-360_dataset/fruits-360/Training"
CATEGORIES = ["Apple Braeburn","Apple Golden 1","Apple Golden 2","Apple Golden 3","Apple Granny Smith",

             "Apple Red 1","Apple Red 2","Apple Red 3","Apple Red Delicious","Apricot","Avocado",

             "Avocado ripe","Banana","Banana Red","Cactus fruit","Cantaloupe 1","Cantaloupe 2","Carambula",

             "Cherry 1","Cherry Rainier","Clementine","Cocos","Dates","Granadilla","Grape Pink","Grape White",

             "Grape White 2","Grapefruit Pink","Grapefruit White","Guava","Huckleberry","Kaki","Kiwi","Kumquats",

             "Lemon","Lemon Meyer","Limes","Lychee","Mandarine","Mango","Maracuja","Nectarine","Orange","Papaya",

             "Passion Fruit","Peach","Peach Flat","Pear","Pear Abate","Pear Monster","Pear Williams","Pepino",

             "Pineapple","Pitahaya Red","Plum","Pomegranate","Quince","Raspberry","Salak","Strawberry",

             "Tamarillo","Tangelo","Melon Piel de Sapo","Strawberry Wedge","Mulberry","Physalis",

             "Physalis with Husk","Rambutan","Cherry Wax Yellow","Cherry Wax Black","Cherry Wax Red","Walnut",

             "Tomato Cherry Red","Tomato Maroon","Tomato 1","Tomato 2","Tomato 3","Tomato 4","Redcurrant",

             "Grape Blue","Cherry 2","Peach 2","Grape White 3","Grape White 4","Apple Red Yellow 1",

             "Apple Red Yellow 2","Banana Lady Finger","Chestnut","Mangostan","Pomelo Sweetie","Hazelnut",

             "Pear Kaiser","Plum 2","Plum 3","Pepper Green","Pepper Red","Pepper Yellow","Apple Crimson Snow",

             "Pear Red"]
data=[]

label=[]

imdir=[]

img1=[]

for category in CATEGORIES:

    path=os.path.join(DATADIR,category)

    imdir.append(path)

    for img in os.listdir(path):

        img_array = cv2.imread(os.path.join(path,img))

        img_array =  cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)

        data.append(img_array)

        label.append(os.path.join(path,img))
from keras.preprocessing import image

from keras.applications.vgg16 import VGG16

from keras.applications.vgg16 import preprocess_input

import numpy as np

from sklearn.cluster import KMeans

import os, shutil, glob, os.path

from PIL import Image as pil_image

image.LOAD_TRUNCATED_IMAGES = True 

model = VGG16(weights='imagenet', include_top=False)
featurelist = []

for i, imagepath in enumerate(label):

    print("    Status: %s / %s" %(i, len(label)), end="\r")

    img = image.load_img(imagepath, target_size=(50, 50))

    img_data = image.img_to_array(img)

    img_data = np.expand_dims(img_data, axis=0)

    img_data = preprocess_input(img_data)

    features = np.array(model.predict(img_data))

    featurelist.append(features.flatten())

    



# Clustering

axi=np.array(featurelist)

number_clusters=10

kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(axi)
kmeans