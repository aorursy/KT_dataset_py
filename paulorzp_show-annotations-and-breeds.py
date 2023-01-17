import numpy as np # linear algebra

import xml.etree.ElementTree as ET # for parsing XML

import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images

import os
print(os.listdir("../input"))
breeds = os.listdir('../input/annotation/Annotation/') # list of all breeds
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(18,16))

for indx, axis in enumerate(axes.flatten()):

    breed = np.random.choice(breeds)

    dog = np.random.choice(os.listdir('../input/annotation/Annotation/' + breed))

    img = Image.open('../input/all-dogs/all-dogs/' + dog + '.jpg') 

    tree = ET.parse('../input/annotation/Annotation/' + breed + '/' + dog)

    root = tree.getroot()

    objects = root.findall('object')

    axis.set_axis_off() 

    imgplot = axis.imshow(img)

    for o in objects:

        bndbox = o.find('bndbox') # reading bound box

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        axis.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # show box

        axis.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # show breed



plt.tight_layout(pad=0, w_pad=0, h_pad=0)