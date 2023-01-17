# This kernel investigates the effect of Image Augmentation
import pandas as pd

import numpy as np

import os 

from glob import glob

import matplotlib.pylab as plt

from skimage.io import imread

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from keras.applications.resnet50 import preprocess_input
# Load data

df = pd.read_csv('../input/Data_Entry_2017.csv')

all_image_paths = {os.path.basename(x): x for x in

                   glob('../input/images*/images/*.png')}

print('Scans found:', len(all_image_paths), ', Total Headers', df.shape[0])

df['Image Index'] = df['Image Index'].map(all_image_paths.get)

df = df.loc[df['Patient Age'] <= 100,:]
# To-do

# Change disease name 

disease_name = 'Infiltration'
# Positive samples

df_pos = df.loc[df["Finding Labels"].str.contains(disease_name),:'View Position']

df_pos[disease_name] = 'Pos'

print("Pos dataframe shape: ",df_pos.shape)
df_pos.head()
# list_img = []

# for i in range(6):

#     img = imread(file_path)

#     list_img.append(img)
# Visualize the original images

num_image = 10

row_num = 2

col_num = num_image // row_num



list_img = []

f, ax = plt.subplots(row_num , col_num, figsize = (30,10))

for i in range(num_image):

    file_path = df_pos.iloc[i,0]

    img_name = file_path.split('/')[-1]

    img = imread(file_path)

    list_img.append(img)

    row = i// col_num

    col = i% col_num

    ax[row,col].imshow(img, cmap='gray')

    ax[row,col].set_title(img_name)

plt.show()
# Define Data Generator

data_generator = ImageDataGenerator(

                    featurewise_center=True,

                    featurewise_std_normalization=True,

#                     preprocessing_function = preprocess_input,

                    samplewise_center=True, 

                    samplewise_std_normalization=True, 

                    horizontal_flip = True,

#                     vertical_flip = False, 

                    height_shift_range= 0.05,

                    width_shift_range=0.1, 

                    rotation_range=5, 

                    shear_range = 0.1,

                    fill_mode = 'constant',

                    cval = 0,                    

                    zoom_range=0.05

                    )
img_arr = np.array(list_img)

img_arr = np.expand_dims(img_arr, axis=-1)



data_generator.fit(img_arr)

imageGen = data_generator.flow(img_arr,batch_size=num_image, shuffle=False)
# After data augmentation

# num_image = 4



f, ax = plt.subplots(num_image,2, figsize = (30,30))



# list_img = []



for i in range(num_image):

    origin_img = list_img[i]

    new_img = imageGen[0][i]

#     row = i/

#     col = i%2   

    ax[i,0].imshow(origin_img[:,:], cmap='gray')

    ax[i,0].set_title("Origin Image")

    ax[i,1].imshow(new_img[:,:,0], cmap='gray')

    ax[i,1].set_title("Augmentation Image")

plt.tight_layout()

plt.show()