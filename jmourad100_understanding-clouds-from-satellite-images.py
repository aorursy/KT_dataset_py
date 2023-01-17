import numpy as np

import pandas as pd

import os, random
! ls '/kaggle/input/understanding_cloud_organization'
Data_dir = "/kaggle/input/understanding_cloud_organization/"



print("There are {} Training images".format(len(os.listdir(Data_dir + 'train_images'))))

print("There are {} Testing images".format(len(os.listdir(Data_dir + 'test_images'))))
train_df = pd.read_csv(Data_dir + "train.csv")

train_df.head()
train_df[['Image', 'Label']] = train_df['Image_Label'].str.split('_', expand=True)

train_df.head()
# There are 4 Lables for each image, but some have EncodedPixels and some don't

train_df.groupby("Image").count().head()
train_df.isnull().sum()

# If a label is present then it's encodedPixels is given, if not it's NaN