import pandas as pd

import numpy as np

import matplotlib.pyplot as plt; import seaborn as sns

plt.style.use('seaborn-whitegrid')

import openslide

import os

import cv2

import torch

train = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv')

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gpu
def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        ax[i//3, i%3].imshow(patch) 

        image.close()       

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)
train.head()
plt.figure(figsize=(10, 7))

sns.countplot(train.data_provider);
plt.figure(figsize=(10, 7))

sns.countplot(train.isup_grade);
plt.figure(figsize=(10, 7))

sns.countplot(train.gleason_score);
from IPython.display import YouTubeVideo

YouTubeVideo("1Q7ERNtLcvk", height=500, width=700)
train[train['data_provider'] == "karolinska"]
train[train['data_provider'] == "radboud"]
plt.figure(figsize=(20, 7))

sns.countplot(train[train['data_provider'] == "radboud"].gleason_score, color="red");

plt.legend()

plt.title("Gleason score(s) of Radboud University's Data");
plt.figure(figsize=(20, 7))

sns.countplot(train[train['data_provider'] == "karolinska"].gleason_score, color="blue");

plt.legend()

plt.title("Gleason score(s) of Karolinska University's Data");
plt.figure(figsize=(20, 7))

sns.countplot(train[train['data_provider'] == "karolinska"].isup_grade, color="blue");

plt.legend()

plt.title("Grade(s) of Karolinska University's Data");
plt.figure(figsize=(20, 7))

sns.countplot(train[train['data_provider'] == "radboud"].isup_grade, color="red");

plt.legend()

plt.title("Grade(s) of Radboud University's Data");
import cv2; fgbg = cv2.createBackgroundSubtractorMOG2()



def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        image = cv2.resize(patch, (256, 256))

        image= fgbg.apply(patch)

        ax[i//3, i%3].imshow(image) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)

def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        image = cv2.resize(patch, (256, 256))

        image=cv2.addWeighted ( image,4, cv2.GaussianBlur( image , (0,0) , 256/10) ,-4 ,128)

        ax[i//3, i%3].imshow(image) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)

def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        image = cv2.resize(patch, (256, 256))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ax[i//3, i%3].imshow(image) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)

def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img

    

def circle_crop(img, sigmaX=10):   

    """

    Create circular crop around image centre    

    """    

    

    img = crop_image_from_gray(img)    

    

    height, width, depth = img.shape    

    

    x = int(width/2)

    y = int(height/2)

    r = np.amin((x,y))

    

    circle_img = np.zeros((height, width), np.uint8)

    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)

    img = cv2.bitwise_and(img, img, mask=circle_img)

    img = crop_image_from_gray(img)

    img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

    return img 



def show_images(df, read_region=(1780,1950)):

    data = df

    f, ax = plt.subplots(3,3, figsize=(16,18))

    for i,data_row in enumerate(data.iterrows()):

        image = str(data_row[1][0])+'.tiff'

        image_path = os.path.join('../input/prostate-cancer-grade-assessment',"train_images",image)

        image = openslide.OpenSlide(image_path)

        spacing = 1 / (float(image.properties['tiff.XResolution']) / 10000)

        patch = image.read_region(read_region, 0, (256, 256))

        patch = np.array(patch)

        image = cv2.resize(patch, (256, 256))

        image = circle_crop(image)

        ax[i//3, i%3].imshow(image) 

        ax[i//3, i%3].axis('off')

        ax[i//3, i%3].set_title(f'ID: {data_row[1][0]}\nSource: {data_row[1][1]} ISUP: {data_row[1][2]} Gleason: {data_row[1][3]}')



    plt.show()

images = [

    '059cbf902c5e42972587c8d17d49efed', '06a0cbd8fd6320ef1aa6f19342af2e68', '06eda4a6faca84e84a781fee2d5f47e1',

    '037504061b9fba71ef6e24c48c6df44d', '035b1edd3d1aeeffc77ce5d248a01a53', '046b35ae95374bfb48cdca8d7c83233f',

    '074c3e01525681a275a42282cd21cbde', '05abe25c883d508ecc15b6e857e59f32', '05f4e9415af9fdabc19109c980daf5ad']   

data_sample = train.loc[train.image_id.isin(images)]

show_images(data_sample)
