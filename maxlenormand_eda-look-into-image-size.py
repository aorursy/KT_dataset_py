import numpy as np

import pandas as pd

import cv2



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import imshow

from PIL import Image



from collections import Counter



%matplotlib inline



import os

print(os.listdir("../input"))
train_labels=pd.read_csv('../input/train.csv', dtype=str)

#Changing the attribute ids into lists instead of str seperated by a ' ' to be able to count them

train_labels['attribute_ids']=train_labels['attribute_ids'].str.split(' ')

test_labels=pd.read_csv('../input/sample_submission.csv', dtype=str)



print('train : \n', train_labels.head())

print('\ntest : \n', test_labels.head())



print('\ntrain shape: ', len(train_labels))

print('\ntest shape: ', len(test_labels))
labels = pd.read_csv('../input/labels.csv', dtype=str)

print('labels : ', '\n', labels.head())



print('\nlabels len :', len(labels))
# Let's show a few images:

for i in range(3):

    name_image=train_labels['id'][i]

    image = plt.imread('../input/train/'+name_image+'.png')

    plt.imshow(image)

    plt.show()
#Let's take a look at the sizes of the images:



width_list = []

height_list = []

for i in range(len(train_labels)):

    name_image=train_labels['id'][i]

    with Image.open('../input/train/'+name_image+'.png') as img:

        width, height = img.size

        #print('width: {} \nheight: {}'.format(width, height))

        width_list.append(width)

        height_list.append(height)

        

average_width = sum(width_list)/len(width_list)

average_height = sum(height_list)/len(height_list)



print('average width: {} and height: {}'.format(average_width, average_height))



fig, ax =plt.subplots(1,2, figsize=(15, 8))



sns.distplot(width_list, ax=ax[0])

ax[0].set_title('Image width')

sns.distplot(height_list, ax=ax[1])

ax[1].set_title('Image height')

fig.show()
image_ratio_list = [int(x)/int(y) for x,y in zip(height_list, width_list)]

mean_ratio = sum(image_ratio_list)/len(image_ratio_list)

print('mean ratio (height/width) of images is: ', mean_ratio)



plt.subplots(figsize=(20, 8))

sns.distplot(image_ratio_list, bins=100)

#plt.axvline(mean_ratio,color='orange', label='mean')

plt.axvline(x=1, color='red', label='x=1')

plt.title('image ratio (height/width) distribution')