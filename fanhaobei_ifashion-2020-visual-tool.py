import pandas as pd

import matplotlib.image as mpimg

import numpy as np

from matplotlib import pyplot as plt

import gc

import cv2
train_df = pd.read_csv('/kaggle/input/imaterialist-fashion-2020-fgvc7/train.csv')
train_df.head()
def rle_to_mask(rle_string,height,width):

    # https://www.kaggle.com/tanreinama/prediction-and-submission-of-attributes

    rows, cols = height, width

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rleNumbers = [int(numstring) for numstring in rle_string.split(' ')]

        rlePairs = np.array(rleNumbers).reshape(-1,2)

        img = np.zeros(rows*cols,dtype=np.uint8)

        for index,length in rlePairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img
def plot_one_mask_GlobalAndLocal(SERIES, figsize=(14, 14) ,alpha = 0.35):

    

    mask = rle_to_mask(SERIES['EncodedPixels'],SERIES['Height'],SERIES['Width'])

    image = cv2.imread("../input/imaterialist-fashion-2020-fgvc7/train/"+str(SERIES['ImageId'])+".jpg")

    b,g,r=cv2.split(image)

    image = cv2.merge([r,g,b])

    

    assert image.shape[0:2] == mask.shape[0:2]

    shape = image.shape[0:2]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].imshow(image)

    ax[0].set_title('ImageId: '+SERIES['ImageId'])

    

    ax[1].imshow(image)

    ax[1].imshow(mask, alpha=alpha) # 重叠 : overlapped

    ax[0].axis('off')

    ax[1].axis('off')

    ax[1].set_title('ClassId: '+str(SERIES['ClassId']))



    image[mask==0] = 255 # 背景为空白 : background is white

    where = np.where(image < 255) # 取掩模最小区域 : minimum mask area 

    if len(where[0]) > 0 and len(where[1]) > 0:

        y1,y2,x1,x2 = min(where[0]),max(where[0]),min(where[1]),max(where[1])

    ax[2].imshow(image[y1:y2,x1:x2])

    ax[2].set_title('AttributesIds: '+SERIES['AttributesIds'])

    

    ax[0].axis('off')

    ax[1].axis('off')    

    ax[2].axis('off')    

    plt.show()

    gc.collect()
def plot_one_ClassID(dataframe,ClassId,nums=10,figsize=(14, 14),alpha = 0.35 ):

    # find

    result = dataframe[dataframe.ClassId == ClassId][0:nums]    

    

    # plot

    for index,ser in result.iterrows():

        plot_one_mask_GlobalAndLocal(ser,figsize,alpha)    
plot_one_ClassID(train_df , 1)
def plot_one_Attribute(dataframe,Attribute,figsize = (14,14),alpha = 0.35):

    # find

    # 筛选出有某个属性的样本 : Filter out samples with a certain attribute

    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.contains

    Attribute = str(Attribute)

    AttributesId_sample = dataframe[dataframe['AttributesIds'].str.contains(Attribute, regex=False,na= False)] # na用来把NaN变为False : Na is used to make Nan false

    

    # 根据ClassId进行分组，每组取一个样本 : Group according to ClassId, one sample for each group.    

    # https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html

    # as_index=False这样ClassId就不会被当成索引 : as_index = False so that ClassId will not be used as an index

    result = AttributesId_sample.groupby(['ClassId'],as_index=False).first() 



    # plot

    for index,ser in result.iterrows():

        plot_one_mask_GlobalAndLocal(ser,figsize,alpha)    

plot_one_Attribute(train_df , 317)