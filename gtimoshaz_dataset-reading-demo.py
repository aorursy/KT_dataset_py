# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xml.etree.ElementTree as ET # for parsing XML



import matplotlib.pyplot as plt # to show images

from PIL import Image # to read images



#from keras.models import Sequential

#from keras.layers import Conv2D, Dense



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
breed_list = os.listdir('../input/annotations/Annotation/') # list of all breeds for further demo
plt.figure(figsize=(10, 10))

for i in range(9):

    plt.subplot(331 + i) # showing 9 random images

    breed = np.random.choice(breed_list) # random breed

    dog = np.random.choice(os.listdir('../input/annotations/Annotation/' + breed)) # random image 

    img = Image.open('../input/images/Images/' + breed + '/' + dog + '.jpg') 

    tree = ET.parse('../input/annotations/Annotation/' + breed + '/' + dog) # init parser for file given

    root = tree.getroot() # idk what's it but it's from documentation

    objects = root.findall('object') # finding all dogs. An array

    plt.imshow(img) # displays photo

    for o in objects:

        bndbox = o.find('bndbox') # reading border coordinates

        xmin = int(bndbox.find('xmin').text)

        ymin = int(bndbox.find('ymin').text)

        xmax = int(bndbox.find('xmax').text)

        ymax = int(bndbox.find('ymax').text)

        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin]) # showing border

        plt.text(xmin, ymin, o.find('name').text, bbox={'ec': None}) # printing breed