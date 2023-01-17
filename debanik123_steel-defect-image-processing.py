import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt 

from PIL import Image

import cv2

import sys

import os 

%matplotlib inline
train_df = pd.read_csv("../input/severstal-steel-defect-detection/train.csv")

#sample_df = pd.read_csv("sample_submission.csv")

train_df
train_df["hasMask"] = ~ train_df.EncodedPixels.isna()

train_df
df_1 = train_df.loc[train_df.ClassId.isin([1])]

df_1.head()

#print(df_1.index)
df_2 = train_df.loc[train_df.ClassId.isin([2])]

df_2.head()
df_2.index
df_3 = train_df.loc[train_df.ClassId.isin([3])]

df_3.head()
df_4 = train_df.loc[train_df.ClassId.isin([4])]

df_4.head()
train_df["class_1"] = pd.Series([i for i in df_1.ImageId])
train_df["class_2"] = pd.Series([i for i in df_2.ImageId])
train_df["class_3"] = pd.Series([i for i in df_3.ImageId])
train_df["class_4"] = pd.Series([i for i in df_4.ImageId])
train_df
train_df.to_csv("submission.csv")
def mask_of_defective_region(img_name, pix_index):

    img = cv2.imread(img_name)

    data = np.array([int(run) for run in train_df.EncodedPixels[pix_index].split(' ')])

    positions = map(int, data[0::2])

    length = map(int, data[1::2])

    mask = np.zeros((256,1600,4),dtype=np.uint8)

    mask_label = np.zeros(1600*256, dtype= np.uint8)

    for p,l in zip(positions, length):

        mask_label[p-1:p+l-1] = 1

    mask[:,:,3] = mask_label.reshape(256,1600,order="F")

    for i in range(4):

        contour,_ = cv2.findContours(mask[:,:,i],cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for i in range(len(contour)):

        p = cv2.polylines(img,contour[i],True,(255,0,0),2)

    #cv2.imshow("img",img)

    #cv2.waitKey(0)

    #cv2.destroyAllWindows()

    return img
def mask_plot(data_frame):

    c = 0

    #plt.subplots(2, 1)

    arr = []

    for index,img in zip(data_frame.index,data_frame.ImageId):

        if c>8:

            break

        #print(index,img)

        img = mask_of_defective_region("../input/severstal-steel-defect-detection/train_images/"+str(img),index)

        plt.imshow(img)

        plt.show()

        arr.append(img)

        c = c+1

    #len(arr)   

    #plt.subplot(nrows, ncols, plot_number) 

    

    

    plt.subplot(331)

    plt.imshow(arr[0])



    plt.subplot(332)

    plt.imshow(arr[1])



    plt.subplot(333)

    plt.imshow(arr[2])



    plt.subplot(334)

    plt.imshow(arr[3])

    

    plt.subplot(335)

    plt.imshow(arr[4])

    

    plt.subplot(336)

    plt.imshow(arr[5])

    

    plt.subplot(337)

    plt.imshow(arr[6])

    

    plt.subplot(338)

    plt.imshow(arr[7])

    

    plt.subplot(339)

    plt.imshow(arr[8])

    

    plt.suptitle('subploting of masked area images')

    plt.show()



    

    
mask_plot(df_4)
def mask_save(data_frame,base_path,child_path):

    try:

        c = 0

        for index,img in zip(data_frame.index,data_frame.ImageId):

            img = mask_of_defective_region("train_images/"+str(img),index)

            cv2.imwrite(base_path+str(child_path)+"/"+ str(c) +".jpg",img)

            c += 1

        print(str(c) +" has been saved")

    except:

        print("Input data's are invalid")

base_path = "cleaning_data_set/train_data/"       

mask_save(None,base_path,None)    

    