import tensorflow as tf

from tensorflow import keras



import os

import pandas as pd

import numpy as np

from skimage.io import imread_collection

import skimage.io

import skimage.color

import skimage.transform

from platform import python_version

import matplotlib.pyplot as plt



print(tf.__version__)

print(python_version())
# extract filenames from the folder of images

filenames = []

for root, dirs, files in os.walk('../input/rsna-hemorrhage-jpg/train_jpg/train_jpg'):

    for file in files:

        if file.endswith('.jpg'):

            filenames.append(file)

            

# should be the same as the images imported

len(filenames)
col_dir = '../input/rsna-hemorrhage-jpg/train_jpg/train_jpg/*.jpg'



# Create a collection with the available images

images = imread_collection(col_dir)

#we could also try what is below,

#this should load the images in the order that we expect, 

#but if automatically alphabetical this isn't necessary:

#images = imread_collection(col_dir, load_pattern = filenames)



#make sure this is equivalent with the number of filenames

len(images)
# Plot the first image

plt.figure()

plt.imshow(images[0])

plt.colorbar()

plt.grid(False)

plt.show()



print(images[0])
# Check shape

print(images[0].shape)

print(images[1].shape)

print(images[2].shape)
# Select only the first 5000 images

images_trn = images[:20000]

print(len(images_trn))

images_val = images[20000:25000]

print(len(images_val))

images_tst = images[25000:30000]

print(len(images_tst))
images_arr_trn = skimage.io.collection.concatenate_images(images_trn)

images_arr_val = skimage.io.collection.concatenate_images(images_val)

images_arr_tst = skimage.io.collection.concatenate_images(images_tst)
# Import labels and selct only first 5000 labels without any additional columns

#labels = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/labels.fth')

#labels = labels.iloc[:5000, 1]

#print(labels)

#print(type(labels))

#print(labels.sum())
labels = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/labels.fth')



#manipulate the filenames list, stripping the .jpg at the end

idstosearch = [item.rstrip(".jpg") for item in filenames]



#now search the "ID" column for ids that correspond to our filenames

#made the reduced dataframe "labels2" for now

labels2 = labels[labels['ID'].isin(idstosearch)]

labels2.shape
labels = labels2.iloc[:, 1]

print(labels)
labels_trn = labels[:20000]

print(len(labels_trn))

labels_val = labels[20000:25000]

print(len(labels_val))

labels_tst = labels[25000:30000]

print(len(labels_tst))
print(type(labels_trn))

print(labels_trn.sum())
# Transform labels into array

labels_trn = pd.Series.to_numpy(labels_trn)

len(labels_trn)
labels_val = pd.Series.to_numpy(labels_val)

len(labels_val)
labels_tst = pd.Series.to_numpy(labels_tst)

len(labels_tst)
# Build the model

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(256, 256, 3)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(23, activation='softmax')

])
# Compile the model

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



# data = train_images.reshape(2000,75,100,1)
# Train model

model.fit(images_arr_trn, labels_trn, epochs=10)
# Validate model

test_loss, test_acc = model.evaluate(images_arr_val, labels_val, verbose=2)



print('\nTest accuracy:', test_acc)
# ToDos:

# 1. Assign labels correctly

# 2. Train / Validation / Test split

# 3. Increase data size

# 4. Use pretrained model to compare
#df_comb = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/comb.fth').set_index('SOPInstanceUID')   

#print(df_comb.shape)



#df_tst = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/df_tst.fth').set_index('SOPInstanceUID')

#print(df_tst.shape)



#df_samp = pd.read_feather('../input/rsna-hemorrhage-jpg/meta/meta/wgt_sample.fth').set_index('SOPInstanceUID')

#print(df_samp.shape)
#from PIL import Image

#import glob



#image_list = []



#for filename in glob.glob('../input/rsna-hemorrhage-jpg/train_jpg/train_jpg/*.jpg'):

#    im=Image.open(filename)

#    image_list.append(im)