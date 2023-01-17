import gc



import numpy as np 

import pandas as pd 



import PIL.Image

import PIL.ImageFont

import PIL.ImageDraw

import PIL.ImageOps
# Path to data 

PATH = '../input/bengaliai-cv19/'
# Train data import

train = pd.read_csv(f'{PATH}train.csv')



# Drop grapheme column

train = train.drop(['grapheme'], axis=1)



# Class labels import

class_map = pd.read_csv(f'{PATH}class_map.csv')
# Train images import (partial)

train_img_0 = pd.read_parquet(f'{PATH}train_image_data_0.parquet')
# Merge train data and images

train_data_0 = pd.merge(train_img_0, train, on='image_id', how='left')

del train_img_0; gc.collect()
print(train_data_0.shape)
print(train_data_0.head())
# Get image with corresponding information

def get_image(train_data, idx):

    image_data = train_data[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].iloc[idx]

    

    image = train_data.iloc[idx].drop(

        ['image_id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).values.astype(np.uint8) 

    

    unpacked_image = PIL.Image.fromarray(image.reshape(137, 236))

    

    return unpacked_image, image_data
# Display image with corresponding data

unpacked_image, image_data = get_image(train_data_0, 20)



print(image_data)
unpacked_image
# Some additional visualization

import matplotlib.pyplot as plt



unpacked_image_1, image_data_1 = get_image(train_data_0, 20)

unpacked_image_2, image_data_2 = get_image(train_data_0, 30)



fig = plt.figure(figsize=(20, 20))



ax_1 = fig.add_subplot(1, 2, 1)

ax_1.imshow(np.asarray(unpacked_image_1), interpolation='nearest', cmap='Greys_r')

ax_1.set_xlim(0, 200)

ax_1.set_ylim(100, 0)

ax_1.text(10, 25, str(image_data_1), bbox={'facecolor': 'white', 'pad': 10})



ax_2 = fig.add_subplot(1, 2, 2)

ax_2.imshow(np.asarray(unpacked_image_2), interpolation='nearest', cmap='Greys_r')

ax_2.set_xlim(0, 200)

ax_2.set_ylim(100, 0)

ax_2.text(10, 25, str(image_data_2), bbox={'facecolor': 'white', 'pad': 10})