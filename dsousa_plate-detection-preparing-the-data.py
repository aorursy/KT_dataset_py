# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import requests

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from PIL import Image

import matplotlib.pyplot as plt

import urllib

import cv2



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import json



# Read the data

data = pd.read_json('../input/Indian_Number_plates.json', lines=True)

pd.set_option('display.max_colwidth', -1)



# Delete the empty column

del data['extras']



# Extract the points of the bounding boxes because thats what we want

data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)



# And drop the rest of the annotation info

del data['annotation']
Images = []

Plates = []



def downloadTraining(df):



    for index, row in df.iterrows():



        # Get the image from the URL

        resp = urllib.request.urlopen(row[0])

        im = np.array(Image.open(resp))



        # We append the image to the training input array

        Images.append(im)  



        # Points of rectangle

        x_point_top = row[1][0]['x']*im.shape[1]

        y_point_top = row[1][0]['y']*im.shape[0]

        x_point_bot = row[1][1]['x']*im.shape[1]

        y_point_bot = row[1][1]['y']*im.shape[0]



        # Cut the plate from the image and use it as output

        carImage = Image.fromarray(im)

        plateImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))

        Plates.append(np.array(plateImage))
downloadTraining(data)
import matplotlib.pyplot as plt

import matplotlib.patches as patches



# Create figure and axes

fig, ax = plt.subplots(2, 1, constrained_layout=True)



# Set title

ax[0].set_title('Input Image')

ax[1].set_title('Output Image')



# Display the images

ax[0].imshow(Images[0])

ax[1].imshow(Plates[0])



plt.show()