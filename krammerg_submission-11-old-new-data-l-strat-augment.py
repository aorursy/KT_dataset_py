import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

import scipy

from tqdm import tqdm

import sys

%matplotlib inline



print(os.listdir("../input"))



#sys.path.append(os.path.abspath('../input/efficientnet/'))

#import efficientnet.keras as efn



sys.path.append(os.path.abspath("../input/efficientnet/efficientnet/")) #kaggle sheet

import efficientnet.keras as efn #kaggle sheet



import seaborn as sns



#IMSIZE = 224 #for Densenet121 / EfficientNetB0

#IMSIZE = 240 #for EfficientNetB1

#IMSIZE = 260 #for EfficientNetB2

IMSIZE = 300 #for EfficientNetB3

#IMSIZE = 380 #for EfficientNetB4



aptos2019_path = "../input/aptos2019-blindness-detection/"

aptos2019_train_path = "../input/aptos2019-blindness-detection/train_images/"

aptos2019_test_path = "../input/aptos2019-blindness-detection/test_images/"

aptos2019_extension = ".png"



#aptos2015_path = "../input/aptos2015-blindness-detection/"

#aptos2015_train_path = "../input/aptos2015-blindness-detection/train_images/"

#aptos2015_test_path = "../input/aptos2015-blindness-detection/test_images/"

aptos2015_path = "../input/aptos2015blindnessdetection/" #kaggles heet

aptos2015_train_path = "../input/aptos2015blindnessdetection/train_images/" #kaggles heet

aptos2015_test_path = "../input/aptos2015blindnessdetection/test_images/" #kaggles heet

aptos2015_extension = ".jpg"



def plot_diag_hist(dataframe, title='NoTitle'):

    f, ax = plt.subplots(figsize=(7, 4))

    ax = sns.countplot(x="diagnosis", data=dataframe, palette="GnBu_d")

    sns.despine()

    plt.title(title)

    plt.show()



trainbase_2019_df = pd.read_csv(aptos2019_path+"train.csv")

test_2019_df = pd.read_csv(aptos2019_path+"test.csv")

print("Shape of 2019 Training Data: {}".format(trainbase_2019_df.shape))

print("Shape of 2019 Test Data: {}\n".format(test_2019_df.shape))

#print("Appearance of 2019 Dataframe:\n{}\n".format(trainbase_2019_df.head()))

#print("Diagnosis Distribution:\n{}\n".format(trainbase_2019_df['diagnosis'].value_counts()/trainbase_2019_df.shape[0]))





trainbase_2015_df = pd.read_csv(aptos2015_path+"train.csv")

test_2015_df = pd.read_csv(aptos2015_path+"test.csv")

print("Shape of 2015 Training Data: {}".format(trainbase_2015_df.shape))

print("Shape of 2015 Test Data: {}\n".format(test_2015_df.shape))

trainbase_2015_df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)

test_2015_df.rename(columns={"image": "id_code", "level": "diagnosis"}, inplace=True)

test_2015_df.drop(columns=["Usage"], inplace=True)



plot_diag_hist(trainbase_2019_df, title="Predictions of 2019 Training Data")

plot_diag_hist(trainbase_2015_df, title="Predictions of 2015 Training Data")

#plot_diag_hist(test_2015_df, title="Predictions of 2015 Test Data")



full_path_train_2019 = lambda id_code : aptos2019_train_path+id_code+aptos2019_extension

full_path_test_2019 = lambda id_code : aptos2019_test_path+id_code+aptos2019_extension

full_path_train_2015 = lambda id_code : aptos2015_train_path+id_code+aptos2015_extension

full_path_test_2015 = lambda id_code : aptos2015_test_path+id_code+aptos2015_extension



trainbase_2019_df["path"] = trainbase_2019_df["id_code"].apply(full_path_train_2019)

test_2019_df["path"] = test_2019_df["id_code"].apply(full_path_test_2019)

trainbase_2015_df["path"] = trainbase_2015_df["id_code"].apply(full_path_train_2015)

test_2015_df["path"] = test_2015_df["id_code"].apply(full_path_test_2015)



#trainbase_2019_df.head()
import imagehash

import psutil

from joblib import Parallel, delayed





def getImageMetaData(file_path):

    with Image.open(file_path) as img:

        img_hash = imagehash.phash(img)

        return img.size, img.mode, str(img_hash), file_path





def getContradictoryImages(datframe):   

    #paralle processing (faster but fills memory)

    #img_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(

    #    (delayed(getImageMetaData)(fp) for fp in df_all["path"]))



    #serial processing

    img_meta_l = []

    for i in tqdm(range(datframe.shape[0])):

        img_meta_l.append(getImageMetaData(datframe.iloc[i]["path"]))



    img_meta_df = pd.DataFrame(np.array(img_meta_l))

    img_meta_df.columns = ['Size', 'Mode', 'Hash', 'path']



    datframe = datframe.merge(img_meta_df, on='path', how='left')

    df_hashgroup = datframe.groupby('Hash').count().reset_index()

    df_hashgroup_dup = df_hashgroup.query('path > 1')

    dup_hash_list = df_hashgroup_dup['Hash'].values

    df_dup = datframe.loc[datframe['Hash'].isin(dup_hash_list)].sort_values('Hash')

    print("{} duplicates found".format(df_dup.shape))

    df_dup["diagnosis_mean"] = np.zeros(df_dup.shape[0])

    for everyhash in dup_hash_list:

        meanvalue = np.mean(df_dup[df_dup["Hash"] == everyhash]["diagnosis"].values)

        df_dup.loc[df_dup["Hash"] == everyhash, ["diagnosis_mean"]] = meanvalue

    df_dup["diagnosis_mean"] = df_dup["diagnosis_mean"] - df_dup["diagnosis"]

    df_dup = df_dup[df_dup["diagnosis_mean"] != 0]

    print("{} contradictory duplicates found".format(df_dup.shape))

    contradictory_ids = df_dup["id_code"]

    return contradictory_ids





#contradictory_2019 = getContradictoryImages(trainbase_2019_df)

contradictory_2019 = ['1632c4311fc9', 'a75bab2463d4', '8273fdb4405e', 'f0098e9d4aee',

 'f066db7a2efe', '278aa860dffd', '8ef2eb8c51c4', '8446826853d0',

 '9a3c03a5ad0f', 'f03d3c4ce7fb', 'b91ef82e723a', '2df07eb5779f',

 '42a850acd2ac', '51131b48f9d4', '8cb6b0efaaac', '0cb14014117d',

 '4a44cc840ebe', 'f7edc074f06b', '9bf060db8376', '4fecf87184e6',

 'e4151feb8443', '46cdc8b685bd', 'd0079cc188e9', '3dbfbc11e105',

 '16ce555748d8', '4d9fc85a8259', 'bacfb1029f6b', 'e12d41e7b221',

 '6165081b9021', '42985aa2e32f', '521d3e264d71', 'fe0fc67c7980',

 'e8d1c6c07cf2', 'f23902998c21', 'b9127e38d9b9', 'e39b627cf648',

 '33778d136069', '4ccfa0b4e96c', '1c9c583c10bf', 'ea15a290eb96',

 '7525ebb3434d', '3cd801ffdbf0', 'c546670d9684', '30cab14951ac',

 'fcc6aa6755e6', '772af553b8b7', '60f15dd68d30', '0243404e8a00',

 '3ddb86eb530e', '7005be54cab1', '3ee4841936ef', '2f284b6a1940',

 'bb5083fae98f', '35aa7f5c2ec0', '1c4d87baaffc', '98f7136d2e7a',

 'e740af6ac6ea', 'f0f89314e860', '14e3f84445f7', '2f7789c1e046',

 'a8e88d4891c4', 'a9e984b57556', '5eb311bcb5f9', '9e3510963315',

 'b187b3c93afb', 'a1b12fdce6c3', '5e7db41b3bee', 'ab50123abadb',

 '80964d8e0863', 'fda39982a810', '0ac436400db4', '8688f3d0fcaf',

 'd85ea1220a03', 'bfefa7344e7d', '6253f23229b1', '76cfe8967f7d',

 'd035c2bd9104', 'cd93a472e5cd']







#contradictory_2015 = getContradictoryImages(trainbase_2015_df)

contradictory_2015 = ['39670_left', '20116_left', '21488_right', '1178_right',

       '38749_right', '18706_right', '20050_left', '24369_left',

       '6200_left', '11496_left', '17616_left', '14481_left',

       '28732_left', '33706_left', '26434_left', '10646_left',

       '22491_left', '2642_right', '16458_right', '30552_right',

       '10734_right', '20187_right', '29023_right', '25064_left',

       '11437_left', '24820_left', '27344_left', '19549_left',

       '12525_left', '12003_right', '35788_right', '16523_right',

       '43803_right', '19071_right', '3026_right', '29246_right',

       '21760_right', '32216_right', '39105_right', '7638_right',

       '997_right', '15254_left', '40674_left', '26111_left',

       '11003_left', '13971_left', '31363_right', '16991_right',

       '42433_right', '24754_right', '15453_right', '35726_right',

       '7271_right', '18476_right', '29986_right', '38481_right',

       '43711_right', '43047_right', '21571_right', '7646_right',

       '16013_right', '24033_right', '20101_right', '16759_right',

       '8187_right', '2379_right', '24318_right', '31729_right',

       '37213_right', '4326_right', '25945_right', '43167_right',

       '13012_right', '19339_right', '10104_right', '17757_right',

       '29312_right', '7250_right', '42873_right', '31849_right',

       '1267_right', '11980_right', '18688_right', '26312_right',

       '21432_right', '1029_right', '23718_right', '12614_right',

       '41134_right', '32112_left', '2229_left', '21077_right',

       '17181_right', '37730_left', '9899_left', '13938_left',

       '32493_left', '12596_left', '20635_left', '32388_left',

       '34701_left', '33173_left', '18304_left', '40704_left',

       '39778_left', '17230_left', '32633_left', '42621_left',

       '42405_left', '15732_right', '18059_right', '41311_left',

       '2500_left', '7179_left', '10808_left', '40198_left', '2010_left',

       '16609_left', '39389_left', '16785_left', '17373_left',

       '2205_left', '33148_left', '9798_left', '27498_left', '10551_left',

       '22118_left', '13264_left', '29740_left', '4676_left', '3059_left',

       '28992_left', '32206_left', '35090_left', '24089_left',

       '40827_left', '2132_left', '18724_left', '34718_left',

       '40058_left', '7315_left', '1607_right', '31709_right',

       '30174_right', '35667_right', '30925_left', '32246_left',

       '41957_left', '28351_left', '5010_left', '11378_left',

       '23531_left', '32807_left', '26383_right', '8889_right',

       '8866_right', '21518_right', '20540_right', '13971_right',

       '27136_right', '8695_right', '7048_right', '20023_left',

       '17629_left', '10003_left', '41514_left', '18950_left',

       '19261_left', '19158_left', '18760_left', '10669_left',

       '23146_left', '23787_left', '37532_left', '15520_left',

       '8621_left', '11375_left', '40001_left', '13811_left',

       '16807_left', '41131_left', '2094_left', '38215_left',

       '26966_left', '28010_left', '13716_left', '34379_left',

       '38324_left', '34780_left', '2812_left', '17739_left',

       '15811_left', '41080_left', '18235_left', '39978_left',

       '37372_left', '8675_left', '14621_left', '11930_left',

       '41012_left', '36021_left', '32098_left', '1988_left', '3464_left',

       '3040_left', '9384_left', '783_left', '18945_left', '8284_left',

       '14987_left', '31000_left', '3754_left', '26051_left',

       '36566_left', '41327_left', '31981_left', '2851_left',

       '22994_left', '42922_left', '4285_left', '11811_left', '9216_left',

       '39863_left', '12208_left', '14160_left', '11409_left',

       '5267_left', '1639_left', '28629_left', '23369_right', '551_right',

       '43440_left', '27440_left', '24828_left', '29894_left',

       '37488_left', '26043_left', '37042_left']







print("Shape of 2019 Training Data: {}".format(trainbase_2019_df.shape))

trainbase_2019_df = trainbase_2019_df[-trainbase_2019_df["id_code"].isin(contradictory_2019)]

print("Shape of 2019 Training Data after removal of bad data: {}".format(trainbase_2019_df.shape))

print("Shape of 2015 Training Data: {}".format(trainbase_2015_df.shape))

trainbase_2015_df = trainbase_2015_df[-trainbase_2015_df["id_code"].isin(contradictory_2015)]

print("Shape of 2015 Training Data after removal of bad data: {}".format(trainbase_2015_df.shape))
#keep all images from classes > 0 and 5000 images from class 0

trainbase_2015_df_reviewed = trainbase_2015_df[trainbase_2015_df["diagnosis"]>0]

temp = trainbase_2015_df[trainbase_2015_df["diagnosis"]==0].sample(5000, replace=True)

trainbase_2015_df_reviewed = pd.concat([trainbase_2015_df_reviewed, temp])

trainbase_2015_df = trainbase_2015_df_reviewed

plot_diag_hist(trainbase_2015_df, title="Predictions of Reviewed 2015 Training Data")
train_raw_df, validation_df = train_test_split(trainbase_2019_df, test_size=0.2, random_state=42, 

                                               stratify=trainbase_2019_df[['diagnosis']])

print("Shape of Training Data: {}".format(train_raw_df.shape))

print("Diagnosis Distribution in Training Set:\n{}\n".format(train_raw_df['diagnosis'].value_counts()/train_raw_df.shape[0]))

print("Shape of Validation Data: {}".format(validation_df.shape))

print("Diagnosis Distribution in Validation Set:\n{}\n".format(validation_df['diagnosis'].value_counts()/validation_df.shape[0]))



train_raw_2015_df, validation_2015_df = train_test_split(trainbase_2015_df, test_size=0.2, random_state=42, 

                                               stratify=trainbase_2015_df[['diagnosis']])

print("Shape of additional 2015 Training Data: {}".format(train_raw_2015_df.shape))

print("Diagnosis Distribution in 2015 Training Set:\n{}\n".format(train_raw_2015_df['diagnosis'].value_counts()/train_raw_2015_df.shape[0]))

print("Shape of additional 2015 Validation Data: {}".format(validation_2015_df.shape))

print("Diagnosis Distribution in 2015 Validation Set:\n{}\n".format(validation_2015_df['diagnosis'].value_counts()/validation_2015_df.shape[0]))





# add all images with classes > 0 from 2015 dataset

#train_df = pd.concat([train_raw_df, train_raw_2015_df[train_raw_2015_df["diagnosis"]>0]], axis=0) 

# align all class sizes to biggest class

#max_size = train_df['diagnosis'].value_counts().max()



#lst = [train_df]

#class_size = [1.0, 1.0, 1.0, 1.0, 1.0]



#for class_index, group in train_df.groupby('diagnosis'):

#    print("Class_index: {}, Is: {}, Should be: {}".format(class_index, group.shape[0], max_size*class_size[class_index]))

#    if (class_index == 0):

#        pass

#        #add data from 2015 set

#        lst.append(train_raw_2015_df[train_raw_2015_df["diagnosis"]==0].sample((max_size*class_size[class_index]).astype(int)-len(group), replace=True))

#    else:

#        pass

#        #duplicate data from mixed 2015 & 2019 set

#        #lst.append(group.sample((max_size*class_size[class_index]).astype(int)-len(group), replace=True))

#train_df = pd.concat(lst)

#train_df = train_raw_df

#train_2015_df = train_raw_2015_df # select only classes > 0

#validation_df = pd.concat([validation_df, validation_2015_df], axis=0) 



train_df = train_raw_df

train_2015_df = train_raw_2015_df



print("Shape of Oversampled Training Data: {}".format(train_df.shape))

print("Shape of Merged Validation Data: {}".format(validation_df.shape))

print("Appearance of Oversampled Training Data:\n{}\n".format(train_df.head()))

print("Diagnosis Distribution in Oversampled Training Set:\n{}\n".format(train_df['diagnosis'].value_counts()/train_df.shape[0]))

plot_diag_hist(train_df, title="Predictions of Stratified and Oversampled Training Data")



train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_2015_df = train_2015_df.sample(frac=1, random_state=42).reset_index(drop=True)

validation_2015_df = validation_2015_df.sample(frac=1, random_state=42).reset_index(drop=True)
import albumentations

import time

import cv2





def show_image(image, figsize=None, title=None):

    if figsize is not None:

        fig = plt.figure(figsize=figsize)

    if image.ndim == 2:

        plt.imshow(image,cmap='gray')

    else:

        plt.imshow(image)

    if title is not None:

        plt.title(title)

        

def show_Nimages(imgs,scale=1):

    N=len(imgs)

    fig = plt.figure(figsize=(25/scale, 16/scale))

    for i, img in enumerate(imgs):

        ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])

        show_image(img)





def augment_aptos(aug, img):

    return aug(image=img)['image']





albumentations



aug_rotate = albumentations.Rotate(p=1, limit=180, interpolation=cv2.INTER_LANCZOS4, 

                    border_mode=cv2.BORDER_CONSTANT,value=0) # value=black

aug_flip = albumentations.Flip(p=0.5)

aug_bright = albumentations.RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45,p=1)

h_min=np.round(IMSIZE*0.8).astype(int)

h_max= np.round(IMSIZE*1.0).astype(int)

aug_crop = albumentations.RandomSizedCrop((h_min, h_max),IMSIZE,IMSIZE, w2h_ratio=1,p=1)

max_hole_size = int(IMSIZE/10)

aug_hole = albumentations.Cutout(p=1, max_h_size=max_hole_size, max_w_size=max_hole_size, num_holes=8 )

aug_flare = albumentations.RandomSunFlare(src_radius=max_hole_size, num_flare_circles_lower=10, num_flare_circles_upper=20, p=1)



full_augmentation = albumentations.Compose([aug_rotate, aug_flip, aug_bright, aug_crop, aug_hole, aug_flare],p=1)







def load_image_preprocess(image_path, black_threshold=20, imsize=IMSIZE, augmentation=False, sub_gauss=True, sub_median=False, zoom_boxed=True):

    img = cv2.imread(image_path)

    #print("Path: {}".format(image_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

    crop_black = True #if false, image is just made quadratic

    

    #get mask from blurred image to find background  assuminig background is black below "black_threshold"

    gray_img = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 5)

    mask = gray_img>black_threshold

        

    if (crop_black):

        if (not(mask.any(axis=0).any())): #image to dark, nothing found

            centerx = int(img.shape[0]//2)

            centery = int(img.shape[1]//2)

            maxdistance = max(img.shape)//2

            border_u_d = int(np.round(maxdistance-centerx))

            border_l_r = int(np.round(maxdistance-centery))

            #print("border: ({}/{})".format(border_u_d, border_l_r))

            #print("Old shape: {}".format(imnew.shape))

            imnew = cv2.copyMakeBorder(img,border_u_d, border_u_d, border_l_r, border_l_r, borderType=cv2.BORDER_CONSTANT, value=0)

            #print("New shape: {}".format(imnew.shape))

        else:

            #get most left/right/upper/lower point that is not background

            xvect = np.where(mask.any(axis=0))[0]

            yvect = np.where(mask.any(axis=1))[0]

            subx = (np.max(xvect) - np.min(xvect)) % 2 #assure even number of pixels

            suby = (np.max(yvect) - np.min(yvect)) % 2 

            #print("Modulos: {} / {}".format(subx, suby))

            #print("{} {} {} {}".format(np.min(yvect), np.max(yvect), np.min(xvect), np.max(xvect)))

            #crop to image content without background

            imnew = img[np.min(yvect):(np.max(yvect)-suby),np.min(xvect):(np.max(xvect)-subx)]

            maskknew = mask[np.min(yvect):(np.max(yvect)-suby),np.min(xvect):(np.max(xvect)-subx)]

            

            centerx = int(imnew.shape[0]/2)

            centery = int(imnew.shape[1]/2)

            #print("center: ({}/{})".format(centerx, centery))

            distance_border = 0

            pointlist = []

            distances = []



            coords = np.where(maskknew[distance_border,:])

            if (coords[0].shape[0] > 0):

                pointlist.append((distance_border, np.min(coords)))

                pointlist.append((distance_border, np.max(coords)))

            coords = np.where(maskknew[imnew.shape[0]-distance_border-1,:])

            if (coords[0].shape[0] > 0):

                pointlist.append((imnew.shape[0]-distance_border-1, np.min(coords)))

                pointlist.append((imnew.shape[0]-distance_border-1, np.max(coords)))

            coords = np.where(maskknew[:,distance_border])

            if (coords[0].shape[0] > 0):

                pointlist.append((np.min(coords), distance_border))

                pointlist.append((np.max(coords), distance_border))

            coords = np.where(maskknew[:,imnew.shape[1]-distance_border-1])

            if (coords[0].shape[0] > 0):

                pointlist.append((np.min(coords), imnew.shape[1]-distance_border-1))

                pointlist.append((np.max(coords), imnew.shape[1]-distance_border-1))

            for xp, yp in pointlist:

                distances.append(((xp-centerx)*(xp-centerx)+(yp-centery)*(yp-centery)))

            maxdistance = int(np.round(np.sqrt(max(distances)))) # retina image radius

            #print("Radius: {}, Diameter: {}".format(maxdistance, maxdistance*2))

            border_u_d = int(np.round(maxdistance-centerx))

            border_l_r = int(np.round(maxdistance-centery))

            #print("border: ({}/{})".format(border_u_d, border_l_r))

            #print("Old shape: {}".format(imnew.shape))

            imnew = cv2.copyMakeBorder(imnew,border_u_d, border_u_d, border_l_r, border_l_r, borderType=cv2.BORDER_CONSTANT, value=0)

            #print("New shape: {}".format(imnew.shape))

    else:

        centerx = int(img.shape[0]//2)

        centery = int(img.shape[1]//2)

        maxdistance = max(img.shape)//2

        border_u_d = int(np.round(maxdistance-centerx))

        border_l_r = int(np.round(maxdistance-centery))

        #print("border: ({}/{})".format(border_u_d, border_l_r))

        #print("Old shape: {}".format(imnew.shape))

        imnew = cv2.copyMakeBorder(img,border_u_d, border_u_d, border_l_r, border_l_r, borderType=cv2.BORDER_CONSTANT, value=0)

        #print("New shape: {}".format(imnew.shape))



    

    

    

    scaling = imsize/imnew.shape[0]

    rotation = 0

    translate_x = 0

    translate_y = 0

    

    if zoom_boxed:

        scaling *= 1.4

    M = cv2.getRotationMatrix2D((maxdistance,maxdistance), rotation, scaling)

    M[0,2] -= (maxdistance - imsize/2 + translate_x)

    M[1,2] -= (maxdistance - imsize/2 + translate_y)

    imnew = cv2.warpAffine(imnew, M, (imsize,imsize))

    

    if sub_gauss:

        k = (imsize//40)*2+1

        bg = cv2.GaussianBlur(imnew ,(0,0) ,k)

        imnew = cv2.addWeighted (imnew, 4, bg, -4, 128)



    if sub_median:

        k = (imsize//40)*2+1

        bg = cv2.medianBlur(imnew, k)

        imnew = cv2.addWeighted (imnew, 4, bg, -4, 128)

    

    if augmentation:

        #Flip H/V

        #if (np.random.rand() > 0.5): #flip h

        #    imnew = cv2.flip(imnew, 0)

        #if (np.random.rand() > 0.5): #flip v

        #    imnew = cv2.flip(imnew, 1)

        #brightness

        #imnew = np.clip((imnew * float(0.75 + np.random.rand()*0.5)),0,255).astype("uint8") #brightness

        #add blur randomly

        #do_blur = np.random.rand()

        #if (do_blur>0.8):

        #    imnew = cv2.GaussianBlur(imnew ,(0,0) ,1)

        #    #print("blurring")

        imnew = augment_aptos(full_augmentation,imnew)

     

    #imnew = (255-imnew) #invert image

    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    #imnew[:,:,1] = clahe.apply(imnew[:,:,1])

    #imnew[:,:,2] = clahe.apply(imnew[:,:,2])

    #imnew[:,:,0] = clahe.apply(imnew[:,:,0])

    

    



        

    return imnew





def display_samples_preprocess_steps(df, columns=3, rows=2):

    fig=plt.figure(figsize=(5*columns, 4*rows))

    for i in range(rows):

        image_path = df.loc[i,"path"]

        image_id = df.loc[i,'diagnosis']

        

        img_0 = load_image_preprocess(image_path,

                                     imsize=IMSIZE,

                                     augmentation=False,

                                     sub_gauss=False,

                                     sub_median=False)

        fig.add_subplot(rows, columns, i*columns + 1)

        plt.title(image_id)

        plt.imshow(img_0)

        

        img_1 = load_image_preprocess(image_path,

                                     imsize=IMSIZE,

                                     augmentation=False,

                                     sub_gauss=True,

                                     sub_median=False)

        fig.add_subplot(rows, columns, i*columns + 2)

        plt.title(image_id)

        plt.imshow(img_1)

        

        img_2 = load_image_preprocess(image_path,

                                     imsize=IMSIZE,

                                     augmentation=False,

                                     sub_gauss=False,

                                     sub_median=True)

        fig.add_subplot(rows, columns, i*columns + 3)

        plt.title(image_id)

        plt.imshow(img_2)

    plt.tight_layout()

    
imagepath = train_df.loc[4,"path"]

img = load_image_preprocess(imagepath, augmentation=False)

#image_path, black_threshold=20, imsize=224, augmentation=False, sub_gauss=False, sub_median=False, zoom_boxed=False

#plt.imshow(seq.augment_image(img))

plt.imshow(img)

print(img.shape)

print((img[np.newaxis]).shape)



display_samples_preprocess_steps(train_df)





#aug_rotate, aug_flip, aug_bright, aug_crop, aug_hole, aug_flare

img1 = augment_aptos(aug_rotate,img)

img2 = augment_aptos(aug_rotate,img)

show_Nimages([img,img1,img2],scale=2)



img1 = augment_aptos(aug_flip,img)

img2 = augment_aptos(aug_flip,img)

img3 = augment_aptos(aug_flip,img)

show_Nimages([img3,img1,img2],scale=2)



img1 = augment_aptos(aug_bright,img)

img2 = augment_aptos(aug_bright,img)

img3 = augment_aptos(aug_bright,img)

show_Nimages([img3,img1,img2],scale=2)



img1 = augment_aptos(aug_crop,img)

img2 = augment_aptos(aug_crop,img)

img3 = augment_aptos(aug_crop,img)

show_Nimages([img3,img1,img2],scale=2)



img1 = augment_aptos(aug_hole,img)

img2 = augment_aptos(aug_hole,img)

img3 = augment_aptos(aug_hole,img)

show_Nimages([img3,img1,img2],scale=2)



img1 = augment_aptos(aug_flare,img)

img2 = augment_aptos(aug_flare,img)

img3 = augment_aptos(aug_flare,img)

show_Nimages([img3,img1,img2],scale=2)



img1 = augment_aptos(full_augmentation,img)

img2 = augment_aptos(full_augmentation,img)

img3 = augment_aptos(full_augmentation,img)

show_Nimages([img3,img1,img2],scale=2)
IS_REGRESSION = False



def extract_target_vector_multi(dataframe_X):

    y_targets = pd.get_dummies(dataframe_X['diagnosis']).values

    y_targets_multi = np.empty(y_targets.shape, dtype=y_targets.dtype)

    y_targets_multi[:, 4] = y_targets[:, 4]

    for i in range(3, -1, -1):

        y_targets_multi[:, i] = np.logical_or(y_targets[:, i], y_targets_multi[:, i+1])

    #use only 4 cols

    #y_train_multi = y_train_multi[:,1:]

    return y_targets_multi



def extract_target_vector_regression(dataframe_X):

    y_targets = dataframe_X['diagnosis'].values

    return y_targets





def result_from_prediction_regression(y_pred):

    return (np.clip(np.round(y_pred),0,4)).astype("uint").flatten()



def result_from_prediction_multi(y_pred):

    return (y_pred > 0.5).astype(int).sum(axis=1) - 1





#datasets are

# trainbase_2019_df / y_trainbase_2019_multi

# train_df / y_train_multi

# validation_df / y_validate_multi

# test_2019_df

#

# trainbase_2015_df / y_trainbase_2015_multi

# train_2015_df

# validation_2015_df

# test_2015_df / y_test_2015_multi



if (IS_REGRESSION):

    y_train_multi = extract_target_vector_regression(train_df)

    y_validate_multi = extract_target_vector_regression(validation_df)

    y_trainbase_2019_multi = extract_target_vector_regression(trainbase_2019_df)

    y_trainbase_2015_multi = extract_target_vector_regression(trainbase_2015_df)

    y_test_2015_multi = extract_target_vector_regression(test_2015_df)

    result_from_prediction = result_from_prediction_regression

else:

    y_train_multi = extract_target_vector_multi(train_df)

    y_validate_multi = extract_target_vector_multi(validation_df)

    y_trainbase_2019_multi = extract_target_vector_multi(trainbase_2019_df)

    y_trainbase_2015_multi = extract_target_vector_multi(trainbase_2015_df)

    y_test_2015_multi = extract_target_vector_multi(test_2015_df)

    result_from_prediction = result_from_prediction_multi





print("Training data:\nXshape: {}\nyshape: {}\n".format(train_df.shape,y_train_multi.shape))

print("Validation data:\nXshape: {}\nyshape: {}\n".format(validation_df.shape,y_validate_multi.shape))

print("Test data:\nXshape: {}\n".format(test_2019_df.shape))

print("Training data multilabel diagnosis vector:", y_train_multi.sum(axis=0))

print("Validation data multilabel diagnosis vector:", y_validate_multi.sum(axis=0))

print("Training data 2015 multilabel diagnosis vector:", y_trainbase_2015_multi.sum(axis=0))

print("Training data 2015 multilabel diagnosis vector:", y_test_2015_multi.sum(axis=0))
from keras.utils import Sequence

from sklearn.utils import shuffle



BATCH_SIZE = 32



class My_Image_Generator(Sequence):



    def __init__(self, image_filenames, labels, batch_size, shuffle=True, augment=False, train=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.is_train = train

        self.batch_size = batch_size

        self.is_shuffle = shuffle

        self.is_augment = augment

        if(self.is_shuffle):

            self.on_epoch_end()

            



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        if (self.is_train):

            batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        else:

            batch_y = np.zeros(self.batch_size)

        return self.batch_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_shuffle):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass



    def batch_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            if(self.is_augment):

                img = load_image_preprocess(sample, augmentation=True)

            else:

                img = load_image_preprocess(sample, augmentation=False)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        #print(batch_images.shape)

        #print(batch_y.shape)

        return batch_images, batch_y

    



#datasets are

# train_df / y_train_multi / train_data_generator_aug / train_data_generator_no_aug

# validation_df / y_validate_multi / valid_data_generator_aug / valid_data_generator_no_aug

# trainbase_2019_df / y_trainbase_2019_multi / trainbase_2019_data_generator_aug / trainbase_2019_data_generator_no_aug

# test_2019_df / test_data_generator_aug / test_data_generator_no_aug

#

# trainbase_2015_df / y_trainbase_2015_multi / trainbase_2015_data_generator_aug / trainbase_2015_data_generator_no_aug

# train_2015_df

# validation_2015_df

# test_2015_df / y_test_2015_multi / test_data_2015_generator_no_aug

    

    

train_data_generator_aug = My_Image_Generator(train_df["path"], labels=y_train_multi, shuffle=True, train=True,

                                              batch_size=BATCH_SIZE, augment=True)

train_data_generator_no_aug = My_Image_Generator(train_df["path"], labels=y_train_multi, shuffle=False, train=True,

                                                 batch_size=BATCH_SIZE, augment=False)

valid_data_generator_aug = My_Image_Generator(validation_df["path"], y_validate_multi, shuffle=False, train=True,

                                              batch_size=BATCH_SIZE, augment=True)

valid_data_generator_no_aug = My_Image_Generator(validation_df["path"], y_validate_multi, shuffle=False, train=True,

                                                 batch_size=BATCH_SIZE, augment=False)





trainbase_2019_data_generator_aug = My_Image_Generator(trainbase_2019_df["path"], labels=y_trainbase_2019_multi, 

                                                       shuffle=True, train=True, batch_size=BATCH_SIZE, augment=True)

trainbase_2019_data_generator_no_aug = My_Image_Generator(trainbase_2019_df["path"], labels=y_trainbase_2019_multi, 

                                                          shuffle=False, train=True, batch_size=BATCH_SIZE, augment=False)

trainbase_2015_data_generator_aug = My_Image_Generator(trainbase_2015_df["path"], labels=y_trainbase_2015_multi, 

                                                       shuffle=True, train=True, batch_size=BATCH_SIZE, augment=True)

trainbase_2015_data_generator_no_aug = My_Image_Generator(trainbase_2015_df["path"], labels=y_trainbase_2015_multi, 

                                                          shuffle=False, train=True, batch_size=BATCH_SIZE, augment=False)





test_data_generator_aug = My_Image_Generator(test_2019_df["path"], labels=None, shuffle=False, train=False,

                                             batch_size=BATCH_SIZE, augment=True)

test_data_generator_no_aug = My_Image_Generator(test_2019_df["path"], labels=None, shuffle=False, train=False,

                                                batch_size=BATCH_SIZE, augment=False)

test_data_2015_generator_no_aug = My_Image_Generator(test_2015_df["path"], labels=y_test_2015_multi, shuffle=False, train=True,

                                                      batch_size=BATCH_SIZE, augment=False)
def show_image(image, figsize=None, title=None):

    if figsize is not None:

        fig = plt.figure(figsize=figsize)

    if image.ndim == 2:

        plt.imshow(image,cmap='gray')

    else:

        plt.imshow(image)

    if title is not None:

        plt.title(title)



def display_generator_images(generator):

    for images, targets in generator:

        N = images.shape[0]

        fig = plt.figure(figsize=(25, 16))

        for i, img in enumerate(images):

            ax = fig.add_subplot(1, N, i + 1, xticks=[], yticks=[])

            show_image(img)

        break



display_generator_images(train_data_generator_aug)

display_generator_images(train_data_generator_no_aug)

display_generator_images(valid_data_generator_aug)

display_generator_images(valid_data_generator_no_aug)

display_generator_images(trainbase_2019_data_generator_aug)

display_generator_images(trainbase_2019_data_generator_no_aug)

display_generator_images(trainbase_2015_data_generator_aug)

display_generator_images(trainbase_2015_data_generator_no_aug)

display_generator_images(test_data_generator_aug)

display_generator_images(test_data_generator_no_aug)

display_generator_images(test_data_2015_generator_no_aug)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping



class QWK_Metrics(Callback):

    def __init__(self, model, validation_generator_X, validation_y):

        self.model_to_save = model

        self.valid_generator = validation_generator_X

        self.valid_y = validation_y

    def on_train_begin(self, logs={}):

        self.val_kappas = [] #init qwk values array

    def on_epoch_end(self, epoch, logs={}):

        y_val_pred = model.predict_generator(generator=self.valid_generator, 

                                             steps=-(self.valid_y.shape[0] // -BATCH_SIZE), 

                                             verbose=1)

        y_val_pred = result_from_prediction(y_val_pred)

        y_val_true = result_from_prediction(self.valid_y)

        _val_kappa = cohen_kappa_score(y_val_pred, y_val_true, weights='quadratic')

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")   

        #if best value, overwrite and save as new best model

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model_to_save.save('model_kappa.h5')

        return

  

    

reduceLR = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=5,

    min_lr=1e-6,

    verbose=1,

    mode='min'

)



earlyStopping = EarlyStopping(

    monitor='val_loss',

    patience=10,

    verbose=1,

    mode='min',

    restore_best_weights=True

)

from keras.utils import multi_gpu_model

from tensorflow.python.client import device_lib

import tensorflow as tf



def get_available_gpus():

    local_device_protos = device_lib.list_local_devices()

    return [x.name for x in local_device_protos if x.device_type == "GPU"]

num_gpu = len(get_available_gpus())

print("Number of available GPUs: {}".format(num_gpu))
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.constraints import maxnorm

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import Dense

from keras import regularizers, optimizers

import tensorflow as tf

from keras.utils import multi_gpu_model

from keras.optimizers import Adam, SGD



#used_init = "he_uniform"

#used_init = "lecun_normal"

#used_init = "golrot_uniform" #standard

#used_init = "golrot_normal"

used_init = "lecun_uniform"



reg_value_kernel = 5e-5

reg_value_bias = 5e-5



def build_model_plainCNN():

    # create model

    model = Sequential()

    model.add(Conv2D(32, (5, 5), input_shape=[224,224,3], activation='relu', kernel_initializer=used_init, 

                     kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias))) # --> 220

    #model.add(GaussianDropout(0.3))

    model.add(MaxPooling2D(pool_size=(2, 2))) # --> 110

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=used_init, 

                     kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias))) # --> 108

    model.add(MaxPooling2D(pool_size=(2, 2))) # -->  54

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=used_init, 

                     kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias))) # --> 52

    model.add(MaxPooling2D(pool_size=(2, 2))) # --> 26

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer=used_init, 

                     kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias))) # --> 24

    model.add(MaxPooling2D(pool_size=(2, 2))) # --> 12

    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer=used_init, 

                     kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias))) #--> 10

    model.add(MaxPooling2D(pool_size=(2, 2))) # --> 5

    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer=used_init, 

                    kernel_regularizer=regularizers.l1(reg_value_kernel), bias_regularizer=regularizers.l1(reg_value_bias)))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='sigmoid', kernel_initializer=used_init))

    

    #model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-3), metrics=['accuracy'])

    

    return model





def build_model_flat():

    # create model

    model = Sequential()

    model.add(Conv2D(32, (9, 9), input_shape=[224,224,3], activation='relu', kernel_initializer=used_init)) # --> 216

    #model.add(GaussianDropout(0.3))

    model.add(MaxPooling2D(pool_size=(4, 4))) # --> 54

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=used_init)) # --> 52

    model.add(MaxPooling2D(pool_size=(4, 4))) # -->  13

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=used_init)) # --> 11

    model.add(MaxPooling2D(pool_size=(4, 4))) # -->  13

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_initializer=used_init))

    model.add(Dropout(0.2))

    model.add(Dense(5, activation='sigmoid', kernel_initializer=used_init))

    

    #model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])

    

    return model







def build_model_densenet121():

    densenet = DenseNet121(weights='../input/densenetpretrained/DenseNet-BC-121-32-no-top.h5', include_top=False, 

                           input_shape=(IMSIZE,IMSIZE,3))

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(densenet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model





def build_model_efficientnetB0():

    effnet = efn.EfficientNetB0(weights=None,

                           include_top=False,

                           input_shape=(IMSIZE, IMSIZE, 3))

    effnet.load_weights("../input/efficientnetpretrained/efficientnet-b0_imagenet_1000_notop.h5")

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(effnet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model

    



def build_model_efficientnetB1(): # Batch Size 16 on Quadro P2000

    effnet = efn.EfficientNetB1(weights=None,

                           include_top=False,

                           input_shape=(IMSIZE, IMSIZE, 3))

    effnet.load_weights("../input/efficientnetpretrained/efficientnet-b1_imagenet_1000_notop.h5")

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(effnet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model

     

def build_model_efficientnetB2(): #requires Batch Size 8 on Quadro P2000

    effnet = efn.EfficientNetB2(weights=None,

                           include_top=False,

                           input_shape=(IMSIZE, IMSIZE, 3))

    effnet.load_weights("../input/efficientnet/pretrainedweights/efficientnet-b2_imagenet_1000_notop.h5")

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(effnet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model



def build_model_efficientnetB3(): #requires Batch Size 8 on Quadro P2000

    effnet = efn.EfficientNetB3(weights=None,

                           include_top=False,

                           input_shape=(IMSIZE, IMSIZE, 3))

    effnet.load_weights("../input/efficientnet/pretrainedweights/pretrainedweights/efficientnet-b3_imagenet_1000_notop.h5")

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    if (IS_REGRESSION):

        ly_dense_2 = layers.Dense(1, activation='linear')

    else:

        ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(effnet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model



def build_model_efficientnetB4(): #requires Batch Size ? on Quadro P2000

    effnet = efn.EfficientNetB4(weights=None,

                           include_top=False,

                           input_shape=(IMSIZE, IMSIZE, 3))

    effnet.load_weights("../input/efficientnet/pretrainedweights/efficientnet-b4_imagenet_1000_notop.h5")

    ly_gap = layers.GlobalAveragePooling2D()

    ly_dropout_1 = layers.Dropout(0.5)

    ly_dense_1 = layers.Dense(256,activation="relu")

    ly_dropout_2 = layers.Dropout(0.5)

    ly_dense_2 = layers.Dense(5, activation='sigmoid')

    model = Sequential()

    model.add(effnet)

    model.add(ly_gap)

    model.add(ly_dropout_1)

    model.add(ly_dense_1)

    model.add(ly_dropout_2)

    model.add(ly_dense_2)

    return model





if (num_gpu == 0):

    #base_model = build_model_flat()

    #base_model = build_model_plainCNN()

    #base_model = build_model_densenet121()

    #base_model = build_model_efficientnetB0()

    #base_model = build_model_efficientnetB2()

    base_model = build_model_efficientnetB3()

    #base_model = build_model_efficientnetB4()

    model = base_model

    print("Training on single CPU... no GPU avaliable")

elif (num_gpu == 1):

    #base_model = build_model_flat()

    #base_model = build_model_plainCNN()

    #base_model = build_model_densenet121()

    #base_model = build_model_efficientnetB0()

    #base_model = build_model_efficientnetB2()

    base_model = build_model_efficientnetB3()

    #base_model = build_model_efficientnetB4()

    model = base_model

    print("Training on single GPU...")

else:

    with tf.device('/cpu:0'):

        #base_model = build_model_flat()

        #base_model = build_model_plainCNN()

        #base_model = build_model_densenet121()

        #base_model = build_model_efficientnetB0()

        #base_model = build_model_efficientnetB2()

        base_model = build_model_efficientnetB3()

        #base_model = build_model_efficientnetB4()

    try:

        model = multi_gpu_model(base_model, gpus=4)

        print("Training on multiple GPUs..")

    except ValueError:

        model = base_model

        print("Training on single GPU or CPU... error when compiling multi_gpu_model...")





#my_optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)



if (IS_REGRESSION):

    model.compile(loss='mse', optimizer=Adam(lr=5e-5), metrics=['accuracy'])

else:

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])    

    

print("Model Summary:")

model.summary()
#datasets are

# train_df / y_train_multi / train_data_generator_aug / train_data_generator_no_aug

# validation_df / y_validate_multi / valid_data_generator_aug / valid_data_generator_no_aug

# trainbase_2019_df / y_trainbase_2019_multi / trainbase_2019_data_generator_aug / trainbase_2019_data_generator_no_aug

# test_2019_df / test_data_generator_aug / test_data_generator_no_aug

#

# trainbase_2015_df / y_trainbase_2015_multi / trainbase_2015_data_generator_aug / trainbase_2015_data_generator_no_aug

# train_2015_df

# validation_2015_df

# test_2015_df / y_test_2015_multi / test_data_2015_generator_no_aug



for layer in model.layers:

    layer.trainable = True

for layer in model.layers[:-5]:

    layer.trainable = False



if (IS_REGRESSION):

    model.compile(loss='mse', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

else:

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-4), metrics=['accuracy']) 



print("Model Summary:")

model.summary()





#train on old 2015 dataset, validate on new dataset until early stopping

kappa_metrics_1 = QWK_Metrics(base_model, trainbase_2019_data_generator_no_aug, y_trainbase_2019_multi) #sets the "not-parallel" for parameter saving in callback function

#train from directory

history_1 = model.fit_generator(

    generator=trainbase_2015_data_generator_aug,

    steps_per_epoch=-(y_trainbase_2015_multi.shape[0] // -BATCH_SIZE),

    #steps_per_epoch=1,

    epochs=5,

    validation_data=trainbase_2019_data_generator_no_aug,

    validation_steps=-(y_trainbase_2019_multi.shape[0] // -BATCH_SIZE),

    #validation_steps=1,

    callbacks=[kappa_metrics_1, reduceLR, earlyStopping]

)



for layer in model.layers:

    layer.trainable = True



if (IS_REGRESSION):

    model.compile(loss='mse', optimizer=Adam(lr=5e-5), metrics=['accuracy'])

else:

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-5), metrics=['accuracy']) 



print("Model Summary:")

model.summary()



kappa_metrics_2 = QWK_Metrics(base_model, trainbase_2019_data_generator_no_aug, y_trainbase_2019_multi) #sets the "not-parallel" for parameter saving in callback function

#train from directory

history_2 = model.fit_generator(

    generator=trainbase_2015_data_generator_aug,

    steps_per_epoch=-(y_trainbase_2015_multi.shape[0] // -BATCH_SIZE),

    #steps_per_epoch=1,

    epochs=5,

    validation_data=trainbase_2019_data_generator_no_aug,

    validation_steps=-(y_trainbase_2019_multi.shape[0] // -BATCH_SIZE),

    #validation_steps=1,

    callbacks=[kappa_metrics_2, reduceLR, earlyStopping]

)



base_model.save('model_phase_1.h5')







if (IS_REGRESSION):

    model.compile(loss='mse', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

else:

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

print("Model Summary:")

model.summary()
#train on new 2019 dataset, validate on new dataset

kappa_metrics_3 = QWK_Metrics(base_model, valid_data_generator_no_aug, y_validate_multi) #sets the "not-parallel" for parameter saving in callback function

#train from directory

history_3 = model.fit_generator(

    generator=train_data_generator_aug,

    steps_per_epoch=-(y_train_multi.shape[0] // -BATCH_SIZE),

    #steps_per_epoch=1,

    epochs=5,

    validation_data=valid_data_generator_no_aug,

    validation_steps=-(y_validate_multi.shape[0] // -BATCH_SIZE),

    #validation_steps=1,

    callbacks=[kappa_metrics_3, reduceLR, earlyStopping]

)



base_model.save('model_phase_2.h5')
def print_learn_results(history, kappa_metrics):

    history_df = pd.DataFrame(history.history)

    f = plt.figure(figsize=(15,5))

    ax = f.add_subplot(131)

    ax.plot(history_df["loss"], label="loss")

    ax.plot(history_df["val_loss"], label = "val_loss")

    ax.legend()

    #history_df[['loss', 'val_loss']].plot()

    ax = f.add_subplot(132)

    ax.plot(history_df["acc"], label="acc")

    ax.plot(history_df["val_acc"], label="val_acc")

    ax.legend()

    #history_df[['acc', 'val_acc']].plot()

    ax = f.add_subplot(133)

    kappa_df = pd.DataFrame(kappa_metrics.val_kappas, columns=["kappa"])

    ax.plot(kappa_df["kappa"], label="kappa")

    ax.legend()

    



print_learn_results(history_1, kappa_metrics_1)

print_learn_results(history_2, kappa_metrics_2)

print_learn_results(history_3, kappa_metrics_3)
kappa_metrics = kappa_metrics_3

history_df = pd.DataFrame(history_3.history)



print("Max Kappa: {}".format(max(kappa_metrics.val_kappas)))

best_epoch = np.argmax(kappa_metrics.val_kappas)+1

print("Max Kappa at epoch {}.".format(best_epoch))

print("Epoch {}: Acc: {:.4f}, ValAcc: {:.4f}, Loss: {:.4f}, ValLoss: {:.4f}".format(best_epoch, 

                                                                    history_df["acc"][best_epoch-1], 

                                                                    history_df["val_acc"][best_epoch-1], 

                                                                    history_df["loss"][best_epoch-1], 

                                                                    history_df["val_loss"][best_epoch-1]

                                                                   ))
base_model.load_weights('model_kappa.h5')

#base_model.load_weights('model_phase_2.h5')
tta_cycles = 5



#test from disk

y_test_tta = model.predict_generator(generator=test_data_generator_no_aug, 

                                       steps=-(test_2019_df.shape[0] // -BATCH_SIZE), 

                                       verbose=1)



# TTA

#for i in tqdm(range(tta_cycles)):

#    y_test_tta += model.predict_generator(generator=test_data_generator_aug,

#                                            steps=-(test_df.shape[0] // -BATCH_SIZE),

#                                            verbose=1)



#print("Test prediction differences: {}".format(np.max(y_test_tta-y_test_tta_2)))  

    

    

#y_test_tta /= tta_cycles+1

y_test_tta = result_from_prediction(y_test_tta)

test_2019_df['diagnosis'] = y_test_tta

test_2019_df.drop(["path"], inplace=True, axis=1)

test_2019_df.to_csv("submission.csv",index=False)

print("Submission complete!")
