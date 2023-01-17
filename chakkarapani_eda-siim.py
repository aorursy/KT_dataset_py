# Import necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pydicom as pdi

import seaborn as sns

import os
BASE_DIR = '/kaggle/input/siim-isic-melanoma-classification/'
train_df = pd.read_csv(BASE_DIR + 'train.csv')

test_df = pd.read_csv(BASE_DIR + 'test.csv')

sample_sub_df = pd.read_csv(BASE_DIR + 'sample_submission.csv')
print("Number of Training Samples : ", train_df.shape[0])

print("Number of Test Samples : ", test_df.shape[0])

print("Number of Training Features : ", train_df.shape[1])

print("Number of Test Features : ", test_df.shape[1])
train_df.head()
test_df.head()
train_df[train_df['target'] == 1].count()
train_df[train_df['target'] == 0].count()
train_df.info()
print("Total number of patients in the training data        : ",train_df['patient_id'].count())

print("Total number of unique patients in the training data : ",train_df['patient_id'].value_counts().shape[0])

print("Total number of patients in the testing data         : ",test_df['patient_id'].count())

print("Total number of unique patients in the testing data  : ",test_df['patient_id'].value_counts().shape[0])
train_df['image_name'] = train_df['image_name'] + '.jpg'

train_df.head()
test_df['image_name'] = train_df['image_name'] + '.jpg'

test_df.head()
# Sample Images on the Negative Class

# Extract numpy values from Image column in data frame

images = train_df[train_df['target'] == 1]['image_name'].values



# Extract 6 random images from it

random_images = [np.random.choice(images) for i in range(6)]



# Location of the image dir

img_dir = BASE_DIR + 'jpeg/train/'



print('Display Random Images')



# Adjust the size of your images

plt.figure(figsize=(20,10))



# Iterate and plot random images

for i in range(6):

    plt.subplot(3, 2, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()    
# Sample Images on the Negative Class

# Extract numpy values from Image column in data frame

images = train_df[train_df['target'] == 0]['image_name'].values



# Extract 6 random images from it

random_images = [np.random.choice(images) for i in range(6)]



# Location of the image dir

img_dir = BASE_DIR + 'jpeg/train/'



print('Display Random Images')



# Adjust the size of your images

plt.figure(figsize=(20,10))



# Iterate and plot random images

for i in range(6):

    plt.subplot(3, 2, i + 1)

    img = plt.imread(os.path.join(img_dir, random_images[i]))

    plt.imshow(img, cmap='gray')

    plt.axis('off')

    

# Adjust subplot parameters to give specified padding

plt.tight_layout()    
# Import data generator from keras

from keras.preprocessing.image import ImageDataGenerator
# Normalize images

image_generator = ImageDataGenerator(

    samplewise_center=True, #Set each sample mean to 0.

    samplewise_std_normalization= True # Divide each input by its standard deviation

)
# Flow from directory with specified batch size and target image size

generator = image_generator.flow_from_dataframe(

        dataframe=train_df,

        directory=BASE_DIR+"jpeg/train/",

        x_col="image_name", # features

        y_col= ['target'], # labels

        class_mode="raw", 

        batch_size= 1, # images per batch

        shuffle=False, # shuffle the rows or not

        target_size=(320,320) # width and height of output image

)
# Plot a processed image

generated_image, label = generator.__getitem__(1)

plt.imshow(generated_image[0])

plt.colorbar()

print(f'Melanoma Image - Label :' , label)

print(f"The dimensions of the image are {generated_image.shape[1]} pixels width and {generated_image.shape[2]} pixels height")

print(f"The maximum pixel value is {generated_image.max():.4f} and the minimum is {generated_image.min():.4f}")

print(f"The mean value of the pixels is {generated_image.mean():.4f} and the standard deviation is {generated_image.std():.4f}")
generated_image.shape