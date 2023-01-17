import os

import numpy as np

import pandas as pd 

from PIL import Image

from cv2 import resize

import matplotlib.pyplot as plt

print(os.listdir("../input"))
# VGG 16 Places 365 scripts in custom dataset

os.chdir("/kaggle/input/keras-vgg16-places365/")

from vgg16_places_365 import VGG16_Places365

os.chdir("/kaggle/working/")
# Get List of Images

image_samples = '../input/google-landmark-2019-samples/'

all_images = os.listdir(image_samples)



# Resize all images

all_images_resized = []

for filename in all_images:    

    im = np.array(Image.open(image_samples + filename).resize((224, 224), Image.LANCZOS))    

    all_images_resized.append(im)
# Plot image examples

fig = plt.figure(figsize = (16, 32))

for index, im in zip(range(1, len(all_images_resized)+1), all_images_resized):

    fig.add_subplot(10, 5, index)

    plt.title(filename)

    plt.imshow(im)   
# Placeholders for predictions

p0, p1, p2 = [], [], []



# Places365 Model

model = VGG16_Places365(weights='places')

topn = 5



# Loop through all images

for image in all_images_resized:

    

    # Predict Top N Image Classes

    image = np.expand_dims(image, 0)

    topn_preds = np.argsort(model.predict(image)[0])[::-1][0:topn]



    p0.append(topn_preds[0])

    p1.append(topn_preds[1])

    p2.append(topn_preds[2])



# Create dataframe for later usage

topn_df = pd.DataFrame()

topn_df['filename'] = np.array(all_images)

topn_df['p0'] = np.array(p0)

topn_df['p1'] = np.array(p1)

topn_df['p2'] = np.array(p2)

topn_df.to_csv('topn_class_numbers.csv', index = False)



# Summary

topn_df.head()
# Read Class number, class name and class indoor/outdoor marker

class_information = pd.read_csv('../input/keras-vgg16-places365/categories_places365_extended.csv')

class_information.head()



# Set Class Labels

for col in ['p0', 'p1', 'p2']:

    topn_df[col + '_label'] = topn_df[col].map(class_information.set_index('class')['label'])

    topn_df[col + '_landmark'] = topn_df[col].map(class_information.set_index('class')['io'].replace({1:'non-landmark', 2:'landmark'}))

topn_df.to_csv('topn_all_info.csv', index = False)



# Summary

topn_df.head()   

# Get 'landmark' images

n = 9

landmark_images =  topn_df[topn_df['p0_landmark'] == 'landmark']['filename']

landmark_indexes = landmark_images[:n].index.values



# Plot image examples

fig = plt.figure(figsize = (16, 16))

for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):

    fig.add_subplot(3, 3, index)

    plt.title(filename)

    plt.imshow(im)
# Get 'non-landmark' images

n = 9

landmark_images =  topn_df[topn_df['p0_landmark'] == 'non-landmark']['filename']

landmark_indexes = landmark_images[:n].index.values



# Plot image examples

fig = plt.figure(figsize = (16, 16))

for index, im in zip(range(1, n+1), [ all_images_resized[i] for i in landmark_indexes]):

    fig.add_subplot(3, 3, index)

    plt.title(filename)

    plt.imshow(im)