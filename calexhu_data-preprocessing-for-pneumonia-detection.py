import tensorflow as tf

tf.__version__
import os

import glob



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mimg

%matplotlib inline

import cv2



from os import listdir, makedirs, getcwd, remove

from os.path import isfile, join, abspath, exists, isdir, expanduser

from PIL import Image

from pathlib import Path

from skimage.io import imread

from skimage.transform import resize
# Input data files are available in the "../input/" directory.

INPUT_PATH = "../input/pneumonia-detection/chest_xray"



# List the files in the input directory.

print(os.listdir(INPUT_PATH))
# list of all the training images

train_normal = Path(INPUT_PATH + '/train/NORMAL').glob('*.jpeg')

train_pneumonia = Path(INPUT_PATH + '/train/PNEUMONIA').glob('*.jpeg')



# ---------------------------------------------------------------

# Train data format in (img_path, label) 

# Labels for [ the normal cases = 0 ] & [the pneumonia cases = 1]

# ---------------------------------------------------------------

normal_data = [(image, 0) for image in train_normal]

pneumonia_data = [(image, 1) for image in train_pneumonia]



train_data = normal_data + pneumonia_data



# Get a pandas dataframe from the data we have in our list 

train_data = pd.DataFrame(train_data, columns=['image', 'label'])



# Checking the dataframe...

train_data.head()
# Checking the dataframe...

train_data.tail()
# Shuffle the data 

train_data = train_data.sample(frac=1., random_state=100).reset_index(drop=True)



# Checking the dataframe...

train_data.head(10)
print(train_data)
# Counts for both classes

count_result = train_data['label'].value_counts()

print('Total : ', len(train_data))

print(count_result)



# Plot the results 

plt.figure(figsize=(8,5))

sns.countplot(x = 'label', data =  train_data)

plt.title('Number of classes', fontsize=16)

plt.xlabel('Class type', fontsize=14)

plt.ylabel('Count', fontsize=14)

plt.xticks(range(len(count_result.index)), 

           ['Normal : 0', 'Pneumonia : 1'], 

           fontsize=14)

plt.show()
fig, ax = plt.subplots(3, 4, figsize=(20,15))

for i, axi in enumerate(ax.flat):

    image = imread(train_data.image[i])

    axi.imshow(image, cmap='bone')

    axi.set_title('Normal' if train_data.label[i] == 0 else 'Pneumonia',

                  fontsize=14)

    axi.set(xticks=[], yticks=[])
train_data.to_numpy().shape
# ---------------------------------------------------------

#  1. Resizing all the images to 224x224 with 3 channels.

#  2. Then, normalize the pixel values.  

# ---------------------------------------------------------

def data_input(dataset):

    # print(dataset.shape)

    for image in dataset:

        im = cv2.imread(str(image))

        im = cv2.resize(im, (224,224))

        if im.shape[2] == 1:

            # np.dstack(): Stack arrays in sequence depth-wise 

            #              (along third axis).

            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html

            im = np.dstack([im, im, im])

        

        # ----------------------------------------------------------

        # cv2.cvtColor(): The function converts an input image 

        #                 from one color space to another. 

        # [Ref.1]: "cvtColor - OpenCV Documentation"

        #     - https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html

        # [Ref.2]: "Python计算机视觉编程- 第十章 OpenCV" 

        #     - https://yongyuan.name/pcvwithpython/chapter10.html

        # ----------------------------------------------------------

        x_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        

        # Normalization

        x_image = x_image.astype(np.float32)/255.

        return x_image
# Import training dataset...

x_train, y_train = ([data_input(train_data.iloc[i][:]) for i in range(len(train_data))], 

                    [train_data.iloc[i][1] for i in range(len(train_data))])



# Convert the list into numpy arrays

x_train = np.array(x_train)

y_train = np.array(y_train)

    

print("Total number of validation examples: ", x_train.shape)

print("Total number of labels:", y_train.shape)
x_train[0]
y_train