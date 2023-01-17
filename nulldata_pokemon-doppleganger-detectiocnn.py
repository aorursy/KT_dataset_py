# please note this package - imagededup was custom-installed in this kernel 

from imagededup.methods import CNN

from imagededup.utils import plot_duplicates

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (15, 10)
cnn = CNN()
image_dir = "../input/pokemon-images-and-types/images/images/"
encodings = cnn.encode_images(image_dir=image_dir)
duplicates = cnn.find_duplicates(encoding_map=encodings, scores = True)
for key, value in duplicates.items():

   if len(value) > 0:

    print(key + ",")

    
plot_duplicates(image_dir=image_dir, 

                duplicate_map=duplicates, 

                filename='cascoon.png')
plot_duplicates(image_dir=image_dir, 

                duplicate_map=duplicates, 

                filename='manaphy.png')
plot_duplicates(image_dir=image_dir, 

                duplicate_map=duplicates, 

                filename='plusle.png')