import numpy as np # linear algebra

import xml.etree.ElementTree as ET # for parsing XML

import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images

import os
print(os.listdir("../input"))
breeds = os.listdir('../input/annotation/Annotation/') # list of all breeds

bboxes=[]

for breed in breeds:

    dogs = os.listdir('../input/annotation/Annotation/' + breed) # list of all dogs

    for dog in dogs:

             

        tree = ET.parse('../input/annotation/Annotation/' + breed + '/' + dog)

        root = tree.getroot()

        objects = root.findall('object')

        for o in objects:

            bndbox = o.find('bndbox') # reading bound box

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

            bboxes.append((xmin, ymin, xmax, ymax))

print(len(bboxes))    

bboxes=[]

for breed in breeds:

    dogs = os.listdir('../input/annotation/Annotation/' + breed) # list of all dogs

    for dog in dogs:

             

        tree = ET.parse('../input/annotation/Annotation/' + breed + '/' + dog)

        root = tree.getroot()

        objects = root.findall('object')

        for o in objects:

            bndbox = o.find('bndbox') # reading bound box

            xmin = int(bndbox.find('xmin').text)

            ymin = int(bndbox.find('ymin').text)

            xmax = int(bndbox.find('xmax').text)

            ymax = int(bndbox.find('ymax').text)

        bboxes.append((xmin, ymin, xmax, ymax))     

print(len(bboxes)) 