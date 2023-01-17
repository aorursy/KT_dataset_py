# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os # Handle working with files and directories
import cv2
import PIL
from PIL import Image
# Allow very large images to load
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import matplotlib.pyplot as plt
pokemon_dir = '../input/pokemon/pokemon'
# Convert an image to a jpeg
def convert_to_jpg(img_path):
    # Convert png to jpeg
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img.load()
        background = Image.new("RGB", img.size, (0,0,0))
        background.paste(img, mask=img.split()[3])
        img = np.array(background)
    else:
        img = img.convert('RGB')
        img = np.array(img)
    
    return img
        
# Resize image to 128x128
def resize_img(img):
    img = cv2.resize(img, (128,128))
    return img

# Normalize pixel values from -1 to 1, important when utilizing NNs
def normalize_img(img):
    img = img / 127.5 - 1
    return img

# Open an image, convert to jpeg, resize if needed
def open_convert(img_path):
    # png
    if img_path[-4:] == '.png':
        img = convert_to_jpg(img_path)
    # jpeg, etc.
    else:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = np.array(img)

        
    # Convert to 128x128
    img = resize_img(img)
    
    # Normalize img
    img = normalize_img(img)
    
    # Return resized img
    return img

# Test
img = open_convert('../input/pokemon/pokemon/Aerodactyl/00000048.png~original')
# img = Image.fromarray(img, 'RGB')
# img.save('my.png')
plt.imshow(img)
plt.show()
# Contain images and labels
images = []
labels = []

# How many images per pokemon to load
images_per_pokemon = 15

# Keep track of current iteration
count = 0
# Iterate through each pokemon folder
for pkmn in os.listdir(pokemon_dir):
    pkmn_dir = os.path.join(pokemon_dir, pkmn)
    
    # Current number of images loaded for this pokemon
    curr_imgs = 0
    
    # Add each image to the list, use most relevant search results
    for img in sorted(os.listdir(pkmn_dir)):
        # Attempt to add image and label to list
        try:
            images.append(open_convert(os.path.join(pkmn_dir, img)))
            labels.append(pkmn)
        # Ignore garbage images
        except (ValueError, OSError):
            continue
        count += 1
        # Some visualization for time spent loading
        if count % 1000 == 0:
            print('Current iteration: ' + str(count))
            
        # Increment num images loaded
        curr_imgs += 1
        if curr_imgs >= images_per_pokemon:
            break
plt.imshow(images[4])
plt.show()