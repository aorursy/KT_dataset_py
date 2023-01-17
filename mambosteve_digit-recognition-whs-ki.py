# import all libraries

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import svm



samples = 8000



# load csv files

test_file = pd.read_csv("../input/test.csv", nrows=samples)

train_file = pd.read_csv("../input/train.csv", nrows=samples)



# print shape of the files

print("Test file set has {0[0]} lines and {0[1]} columns".format(test_file.shape))

print("Training file set has {0[0]} lines and {0[1]} columns".format(train_file.shape))
# define function to draw image

def draw(data, color = None):

    plt.imshow(data.as_matrix().reshape((28,28)), cmap = color)

    plt.axis('off')

    

# define function to draw all images

def draw_all(data, color = None):

    num_cols = 3

    num_rows = data.shape[0] / num_cols + 1

    

    plt.figure(figsize = (8,7))

    for i in range(data.shape[0]):

        plt.subplot(num_rows, num_cols, i + 1)

        plt.imshow(data.iloc[i].as_matrix().reshape((28,28)), cmap = color)

        plt.axis('off')

    plt.tight_layout()
# split file into label and pixel data

images = train_file.loc[:,'pixel0':]

labels = train_file.loc[:,'label']



# split training data into training and test data

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)



# print shape of the new data sets

print("Test set has {0[0]} lines and {0[1]} columns".format(test_images.shape))

print("Training set has {0[0]} lines and {0[1]} columns".format(train_images.shape))



# plot sample image

draw_all(train_images.iloc[0:12])
split = 75



test_images -= split

train_images -= split



test_images = test_images.clip_upper(1)

train_images = train_images.clip_upper(1)



test_images = test_images.clip_lower(0)

train_images = train_images.clip_lower(0)



draw_all(train_images.iloc[0:12])
ref_images = pd.DataFrame(train_images.head(0), index = range(10))

ref_images = ref_images.fillna(value = 0)



# create reference images by adding each pixel that is set in the training set pixels

for i in range(train_images.shape[0]):

    ref_images.loc[train_labels.iloc[i]] = ref_images.loc[train_labels.iloc[i]] + train_images.iloc[i]
# scale all values to a value between 0 to 1

for i in range(ref_images.shape[0]):

    i_min = ref_images.iloc[i].min()

    i_max = ref_images.iloc[i].max()

    ref_images.iloc[i] = (ref_images.iloc[i] - i_min) / (i_max - i_min)

    #ref_images.iloc[i] = ref_images.iloc[i].clip_lower(0.5)
# draw reference images

draw_all(ref_images, "jet")
# function for testing a test image against all 10 reference images

def get_digit(source):

    harmony = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for digit in range(len(harmony)):

        harmony[digit] = (source * ref_images.iloc[digit]).mean()

    return harmony.index(max(harmony))
good = 0

for i in range(test_images.shape[0]):

    if test_labels.iloc[i] == get_digit(test_images.iloc[i]):

        good += 1

        

print("Error was {}".format(1 - (good / test_images.shape[0])))