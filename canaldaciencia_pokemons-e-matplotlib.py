# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

pil_im = Image.open('../input/logocanal/LOGO PNG.png')

pil_im
import matplotlib.pyplot as plt
imagem=plt.imread('/kaggle/input/pokemon-images-and-types/images/images/abomasnow.png')
plt.imshow(imagem)

plt.show()
plt.figure(figsize=(10,10))

plt.imshow(imagem)

plt.show()
mypath='/kaggle/input/pokemon-images-and-types/images/images/'



from os import listdir

from os.path import isfile, join

pokemon_images = [mypath+f for f in listdir(mypath) if isfile(join(mypath, f))]
pokemon_images[0]
img=plt.imread(pokemon_images[0])

plt.imshow(img)

plt.show()
plt.figure(figsize=(100,100))

plt.subplots_adjust(left=0.1, right=0.2, top=0.2, bottom=0.1)



# generate samples and plot

for i in range(9):

    plt.subplot(330 + 1 + i)

    img=plt.imread(pokemon_images[i])

    plt.imshow(img)

    plt.axis('off')

# show the figure

plt.show()