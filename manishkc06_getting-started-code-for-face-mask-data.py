import pandas as pd                                     # Data analysis and manipultion tool

import numpy as np                                      # Fundamental package for linear algebra and multidimensional arrays

import tensorflow as tf                                 # Deep Learning Tool

import os                                               # OS module in Python provides a way of using operating system dependent functionality

import cv2                                              # Library for image processing

from sklearn.model_selection import train_test_split    # For splitting the data into train and validation set
labels = pd.read_csv("../input/face-mask-dataset/train_labels.csv")   # loading the labels

labels.head()           # will display the first five rows in labels dataframe
labels.tail()            # will display the last five rows in labels dataframe
file_paths = [[fname, '/kaggle/input/face-mask-dataset/train/train/' + fname] for fname in labels['filename']]

file_paths
# Confirm if number of images is same as number of labels given

if len(labels) == len(file_paths):

    print('Number of labels i.e. ', len(labels), 'matches the number of filenames i.e. ', len(file_paths))

else:

    print('Number of labels does not match the number of filenames')
#viewing any image from the train data.

from IPython.display import Image

Image('/kaggle/input/face-mask-dataset/train/train/Image_1000.jpg')
images = pd.DataFrame(file_paths, columns=['filename', 'filepaths'])

images.head()
train_data = pd.merge(images, labels, how = 'inner', on = 'filename')

train_data.head()       
data = []     # initialize an empty numpy array

image_size = 100      # image size taken is 100 here. one can take other size too

for i in range(len(train_data)):





    img_array = cv2.imread(train_data['filepaths'][i], cv2.IMREAD_GRAYSCALE)   # converting the image to gray scale



    new_img_array = cv2.resize(img_array, (image_size, image_size))      # resizing the image array



    # encoding the labels. with_mask = 1 and without_mask = 0

    if train_data['label'][i] == 'with_mask':

        data.append([new_img_array, 1])

    else:

        data.append([new_img_array, 0])
# image pixels of a image

data[0]
# The shape of an image array

data = np.array(data)

data[0][0].shape
np.random.shuffle(data)
import matplotlib.pyplot as plt
# code to view the images

num_rows, num_cols = 2, 5

f, ax = plt.subplots(num_rows, num_cols, figsize=(12,5),

                     gridspec_kw={'wspace':0.03, 'hspace':0.01}, 

                     squeeze=True)



for r in range(num_rows):

    for c in range(num_cols):

      

        image_index = r * 100 + c

        ax[r,c].axis("off")

        ax[r,c].imshow( data[image_index][0], cmap='gray')

        if data[image_index][1] == 0:

          ax[r,c].set_title('without_mask')

        else:

          ax[r,c].set_title('with_mask')

plt.show()

plt.close()
x = []

y = []

for image in data:

  x.append(image[0])

  y.append(image[1])



# converting x & y to numpy array as they are list

x = np.array(x)

y = np.array(y)
np.unique(y, return_counts=True)
x = x / 255



# Why divided by 255?

# --> The pixel value lie in the range 0 - 255 representing the RGB (Red Green Blue) value.
# split the data

X_train, X_val, y_train, y_val = train_test_split(x,y,test_size=0.3, random_state = 42)



# X_train: independent/input feature data for training the model

# y_train: dependent/output feature data for training the model

# X_test: independent/input feature data for testing the model; will be used to predict the output values

# y_test: original dependent/output values of X_test; We will compare this values with our predicted values to check the performance of our built model.

 

# test_size = 0.30: 30% of the data will go for test set and 70% of the data will go for train set

# random_state = 42: this will fix the split i.e. there will be same split for each time you run the co
# Defining the model

model = tf.keras.Sequential([

    tf.keras.layers.Flatten(input_shape=(100, 100)),    # flattening the image

    tf.keras.layers.Dense(100, activation='relu'),

    tf.keras.layers.Dense(50, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10, batch_size = 20)
model.evaluate(X_val, y_val)