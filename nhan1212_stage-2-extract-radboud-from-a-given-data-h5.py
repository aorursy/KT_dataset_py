import os
import cv2
import openslide
import skimage.io
import matplotlib
import numpy as np 
import pandas as pd 
import deepdish as dd
import seaborn as sns
import matplotlib.pyplot as plt

## print out the names of the first 3 image_files (total = 10 images for train_imgaes & train_label_masks) with the train, test, submission.csv files
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))
BASE_PATH = '../input/prostate-cancer-grade-assessment'

data_dir = f'{BASE_PATH}/train_images'
mask_dir = f'{BASE_PATH}/train_label_masks'
h5_dir = '/kaggle/input/h5-files/full_data_coordinate.h5'

train = pd.read_csv(f'{BASE_PATH}/train.csv').set_index('image_id')
test = pd.read_csv(f'{BASE_PATH}/test.csv')
submission = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
train.head(5)
def plot_relative_distribution(df, feature, hue, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(10*size, 4*size))
    total = float(len(df))
    sns.countplot(x=feature, hue=hue, data=df, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2,  ## set the text in the middle of each column 
                height*0.7,                   ## set the text above 0.8*height of the column's height
                '{:d} img_id \n\n approx {:1.2f}%'.format(height, (height*100/total)), ## content
                ha="center") 
    plt.show()
    
plot_relative_distribution(df=train, feature='isup_grade', hue='data_provider', title = 'relative count plot of isup_grade with data_provider', size=2)
def plot_count(df, feature, title='', size=2):
    f, ax = plt.subplots(1,1, figsize=(5.5*size, 3.25*size))
    total = float(len(df))
    sns.countplot(df[feature],order = df[feature].value_counts().index, palette='Set2')
    plt.title(title)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2,  ## set the text in the middle of each column 
                height*0.8,                   ## set the text above 0.8*height of the column's height
                '{:d} img_id \n\n approx {:1.2f}%'.format(height, (height*100/total)), ## content
                ha="center") 
    plt.show()
    
plot_count(df=train, feature='data_provider', title = 'data_provider count_plot')
%time df = dd.io.load(h5_dir)
len(df), df[0]
def display_tiles_in_data(img_ID, level = 0):
    idx = df[img_ID][0]
    im = skimage.io.MultiImage(os.path.join(data_dir, f'{idx}.tiff'))[level]
    plt.figure(figsize = (25, 25))
    for k in range(36):
        plt.subplot(6, 6, k + 1)
        a, b, c, d = df[img_ID + k][1 :]
        plt.imshow(im[a : b, c: d, :])
        plt.title(k)
        
display_tiles_in_data(720)
def display_tiles_in_mask(img_ID, level = 0):
    idx = df[img_ID][0]
    im = skimage.io.MultiImage(os.path.join(mask_dir, f'{idx}_mask.tiff'))[0]
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    plt.figure(figsize = (25, 25))
    for k in range(36):
        plt.subplot(6, 6, k + 1)
        a, b, c, d = df[img_ID + k][1 :]
        plt.imshow(im[a : b, c: d, 0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        plt.title(k)
        
display_tiles_in_mask(720)
def display_tiles_df_h5_file(img_ID, tile_indexes, level = 0):
    """
    Load & display the indexes of {image_id and tiles_coordinates} in h5.file
    Input args:
        img_ID (int : a multiple of 36, less than len of h5.tile): img_id from the data_dir 
        tile_indexes (list of int from [0, 36]): list of the indexes of 36 tiles per image from data_dir
    output:
        display mask_img & data_img coresponding to img_id and its indexes
    """
    img_id = df[img_ID][0]
    img_data = skimage.io.MultiImage(os.path.join(data_dir, f'{img_id}.tiff'))[level]
    img_mask = skimage.io.MultiImage(os.path.join(mask_dir, f'{img_id}_mask.tiff'))[level]
    cmap = matplotlib.colors.ListedColormap(['black', 'gray', 'green', 'yellow', 'orange', 'red'])
    nrow = len(tile_indexes)
    resize = 2**(2*level)
    plt.figure(figsize = (10, 5.5*nrow))
    for k in range(nrow):
        coordinates = df[img_ID + tile_indexes[k]][1 : ]
        x_start, x_end, y_start, y_end = [int(coordinates[j] / resize) for j in range(4)]
        plt.subplot(nrow, 2, 2*(k+1) -1 )
        plt.imshow(img_data[x_start: x_end, y_start: y_end, :])
        plt.title('ID = %s \n level = %s & tile_index = %s'%(img_id, level, tile_indexes[k]))
        plt.subplot(nrow, 2, 2*(k+1))
        plt.imshow(img_mask[x_start: x_end, y_start: y_end,0], cmap=cmap, interpolation='nearest', vmin=0, vmax=5)
        plt.title('mask image')
        
display_tiles_df_h5_file(1080, [15, 7, 12, 21, 28])
DF = []
for idx in range(len(df)):
    img_id = df[idx][0]
    data_provider = train.loc[img_id, 'data_provider']
    if data_provider == 'radboud':
        DF += [df[idx]]
DF[0], DF[-1], len(DF)
%time dd.io.save('radboud_tiles_coordinates.h5', data = DF)
len(DF)
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device