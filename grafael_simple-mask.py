# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Dataset Paths

train_path = '../input/train_images/'

test_path = '../input/test_images/'



# Loading train dataset

train = pd.read_csv("../input/train.csv")
# Removing NaN values

train = train[pd.notnull(train['EncodedPixels'])]
# ImageId column

train['ImageId'] = train['ImageId_ClassId'].apply(lambda x: x.split('_')[0])



# Creating a class column based on classId

train['ClassId'] = train['ImageId_ClassId'].apply(lambda x: x.split('_')[1]).astype(int)
def create_mask(idx, df):



    # Load the encoded pixels

    pixels = df['EncodedPixels'].iloc[idx]

    pixels = pixels.split(" ")

    

    # Get the positions

    positions = map(int, pixels[0::2])

    # Get the length

    length = map(int, pixels[1::2])

 

    # Create an empty flat array 

    mask = np.zeros(256*1600, dtype=np.uint8)

    

    # Set as '1' the array positions 

    for pos, le in zip(positions, length):

        mask[pos:(pos+le)] = 1

         

    # Reshape from flat to image shape.

    mask = mask.reshape(256, 1600, order='F')

    

    return mask
def draw_mask(img, idx, df):

    

    # Create mask data

    mask = create_mask(idx, df)



    # Transparency factor.

    alpha = 0.7

    

    # Pallete (classId:(r,g,b))

    classId_colormap = {

        1: (249,38,114),

        2: (166,226,46),

        3: (102,217,239),

        4: (174,129,255)

    }

    

    # Label color

    classId  = df['ClassId'].iloc[idx]

    

    # Get the line color based on classId

    color = classId_colormap.get(classId, (0,0,0))

    

    # Image copy to apply transparency.

    overlay = img.copy()



    # Get contours from mask

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



    # Create overlay polygon

    mask_img = cv2.fillPoly(overlay, contours, color)



    # Merge image and overlay

    final = cv2.addWeighted(mask_img, alpha, img, 1 , 0)



    return final



    

def show_image(idx):

    

    fig, ax = plt.subplots(figsize=(15, 15))

    

    imageId = train['ImageId'].iloc[idx]

    image_path = '{}/{}'.format(train_path, imageId)

    

    # Load image

    img = cv2.imread(image_path)

    

    # Apply mask 

    img = draw_mask(img, idx, train)

    

    ax.set_title(image_path)

    ax.imshow(img)

    plt.show()
for idx in range(5):

    show_image(idx)