import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import os

import cv2



import collections

import time 

import tqdm

from PIL import Image

from functools import partial

train_on_gpu = True

from matplotlib.pyplot import imread

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import skimage

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from matplotlib.colors import LinearSegmentedColormap

from skimage.util import img_as_float













# first we inspect the data given



# make table 

train_df = pd.read_csv('../input/understanding_cloud_organization/train.csv')

train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

train_df['ClassId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

#train_df['EncodedPixels'] = train_df['EncodedPixels'].apply(lambda x: x.split(' ')[0])

train_df['hasMask'] = ~ train_df['EncodedPixels'].isna()



#train_df.head(n=10)

#print(train_df)





























# here i just define the image splitter function aka a cleaned up version of the visualization 





def rle_decode(mask_rle, shape=(1400, 2100)):

    

    try: # label might not be there!

        s = mask_rle.split()

        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        starts -= 1

        ends = starts + lengths

        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

        for lo, hi in zip(starts, ends):

            img[lo:hi] = 1

        return img.reshape(shape, order='F')  # Needed to align to RLE direction

    except:

        return np.zeros((1400, 2100))



def my_process(image):

    kernel = np.ones((51,51),np.uint8)

    erosion = cv2.erode(image,kernel,iterations = 1)

    dilation = cv2.dilate(erosion,kernel,iterations = 1)

    return dilation

   

def decode_image_data(image_name):



    # decode the base masks into image masks

    label = 'Fish'

    image_label = image_name + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    Fish = rle_decode(mask_rle)

    

    label = 'Flower'

    image_label = image_name + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    Flower = rle_decode(mask_rle)

    

    label = 'Gravel'

    image_label = image_name + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    Gravel = rle_decode(mask_rle)

    

    label = 'Sugar'

    image_label = image_name + '_' + label

    mask_rle = train_df.loc[train_df['Image_Label'] == image_label, 'EncodedPixels'].values[0]

    Sugar = rle_decode(mask_rle)





    

    Fish = Fish.astype('uint8')

    Flower = Flower.astype('uint8')

    Gravel = Gravel.astype('uint8')

    Sugar = Sugar.astype('uint8')

    

    #print(Fish.dtype)

    #print(Flower.dtype)

    #print(Gravel.dtype)

    #print(Sugar.dtype)

    

    # get the image, this is needed to find the part of the image the satelite did not take image of

    imgcv2 = cv2.imread('../input/understanding_cloud_organization/train_images/' + image_name,0)



    # define the combination images

    Fish_and_Flower = Fish & Flower    

    Flower_and_Gravel = Flower & Gravel

    Gravel_and_Sugar = Gravel & Sugar

    Flower_and_Sugar = Flower & Sugar

    Gravel_and_Fish = Gravel & Fish

    Fish_and_Sugar = Fish & Sugar

    Fish_and_Sugar_and_Gravel = Fish & Sugar & Gravel

    Fish_and_Sugar_and_Flower = Fish & Sugar & Flower

    Fish_and_Gravel_and_Flower = Fish & Gravel & Flower

    Sugar_and_Gravel_and_Flower = Sugar & Gravel & Flower

    Only_Fish   = cv2.subtract(Fish,Sugar)

    Only_Fish   = cv2.subtract(Only_Fish,Gravel)

    Only_Fish   = cv2.subtract(Only_Fish,Flower)

    Only_Fish = np.clip(Only_Fish, a_min = 0, a_max = 1)

    Only_Flower   = cv2.subtract(Flower,Fish)

    Only_Flower   = cv2.subtract(Only_Flower,Sugar)

    Only_Flower   = cv2.subtract(Only_Flower,Gravel)

    Only_Flower = np.clip(Only_Flower, a_min = 0, a_max = 1)

    Only_Gravel   = cv2.subtract(Gravel,Fish)

    Only_Gravel   = cv2.subtract(Only_Gravel,Sugar)

    Only_Gravel   = cv2.subtract(Only_Gravel,Flower)

    Only_Gravel = np.clip(Only_Gravel, a_min = 0, a_max = 1)

    Only_Sugar   = cv2.subtract(Sugar,Fish)

    Only_Sugar   = cv2.subtract(Only_Sugar,Gravel)

    Only_Sugar   = cv2.subtract(Only_Sugar,Flower)

    Only_Sugar = np.clip(Only_Sugar, a_min = 0, a_max = 1)

    Black = np.zeros((1400, 2100))

    Black = Black.astype('uint8')

    ret,Black = cv2.threshold(imgcv2,0,5,cv2.THRESH_BINARY_INV)

    Black = cv2.medianBlur(Black, 3)

    kernel = np.ones((5,5), np.uint8) 

    Black = cv2.dilate(Black, kernel, iterations=1)

    Black = np.clip(Black, a_min = 0, a_max = 1)

    Fish_and_Flower = my_process(Fish_and_Flower)

    Flower_and_Gravel = my_process(Flower_and_Gravel)

    Gravel_and_Sugar = my_process(Gravel_and_Sugar)

    Flower_and_Sugar = my_process(Flower_and_Sugar)

    Gravel_and_Fish = my_process(Gravel_and_Fish)

    Fish_and_Sugar = my_process(Fish_and_Sugar)

    Fish_and_Sugar_and_Gravel = my_process(Fish_and_Sugar_and_Gravel)

    Fish_and_Sugar_and_Flower = my_process(Fish_and_Sugar_and_Flower)

    Fish_and_Gravel_and_Flower = my_process(Fish_and_Gravel_and_Flower)

    Sugar_and_Gravel_and_Flower = my_process(Sugar_and_Gravel_and_Flower)

    Only_Sugar = my_process(Only_Sugar)

    Only_Flower = my_process(Only_Flower)

    Only_Gravel = my_process(Only_Gravel)

    Only_Sugar = my_process(Only_Sugar)

    Noclass =  np.ones((1400, 2100))*255

    classs = +Fish | +Flower | +Gravel | +Sugar | +Black

    ret,classs = cv2.threshold(classs,0,255,cv2.THRESH_BINARY)

    Noclass = Noclass.astype('uint8') - classs

    Noclass = np.clip(Noclass, a_min = 0, a_max = 1)

    Noclass = my_process(Noclass)



    # returned in order of purity

    return Only_Fish, Only_Flower, Only_Gravel, Only_Sugar, Fish_and_Flower, Flower_and_Gravel, Gravel_and_Sugar, Flower_and_Sugar, Gravel_and_Fish, Fish_and_Sugar, Fish_and_Sugar_and_Gravel, Fish_and_Sugar_and_Flower, Fish_and_Gravel_and_Flower, Sugar_and_Gravel_and_Flower, Black, Noclass 



#method test 
C1,C2,C3,C4, C5,C6,C7,C8,C9,C10, C11,C12,C13,C14, C15,C16 =  decode_image_data('002be4f.jpg')



fig = plt.figure(figsize=(25/2, 16/2))

#background - Black

total =  C15*0 + C16*0

#pure - white

total = total + C1*255 + C2*255 + C3*255 + C4*255

#2 intersection light gray

total = total + C5 * 175 + C6 * 175 + C7 * 175 + C8 * 175 + C9 * 175 + C10 * 100

#3 intersection dark gray

total = total + C11 * 100 + C12 * 100 + C13 * 100 + C14 *100



plt.imshow(total, alpha=1, cmap='gray')

plt.show()
import time



def getEncodedPixels(image):

    """

    receives a masked image and encodes it to RLE

    :param mask_image:

    :return: string corresponding to the rle of the input image

    """

    pixels = image.flatten()

    # We avoid issues with '1' at the start or end (at the corners of

    # the original image) by setting those pixels to '0' explicitly.

    # We do not expect these to be non-zero for an accurate mask,

    # so this should not harm the score.

    pixels[0] = 0

    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2

    runs[1::2] = runs[1::2] - runs[:-1:2]

    return ' '.join(str(x) for x in runs)









def save_line_to_csv_file(name,pixels):

    mylist = [ name , pixels ]

    wr.writerow(mylist)



import csv



with open('trainC1-16.csv', 'w') as myfile:

    wr = csv.writer(myfile, quoting = csv.QUOTE_NONE)



    save_line_to_csv_file( 'Image_Label'  , 'EncodedPixels' )

    

    for file_id in range(0, 5546):

        print(file_id)

        image_real_name = train_df.loc[file_id*4,'ImageId']

        print(image_real_name)

        C1,C2,C3,C4, C5,C6,C7,C8,C9,C10, C11,C12,C13,C14, C15,C16 = decode_image_data(image_real_name)

        save_line_to_csv_file( image_real_name + '_' + 'C1'  , getEncodedPixels(C1) )

        save_line_to_csv_file( image_real_name + '_' + 'C2'  , getEncodedPixels(C2) )

        save_line_to_csv_file( image_real_name + '_' + 'C3'  , getEncodedPixels(C3) )

        save_line_to_csv_file( image_real_name + '_' + 'C4'  , getEncodedPixels(C4) )

        save_line_to_csv_file( image_real_name + '_' + 'C5'  , getEncodedPixels(C5) )

        save_line_to_csv_file( image_real_name + '_' + 'C6'  , getEncodedPixels(C6) )

        save_line_to_csv_file( image_real_name + '_' + 'C7'  , getEncodedPixels(C7) )

        save_line_to_csv_file( image_real_name + '_' + 'C8'  , getEncodedPixels(C8) )

        save_line_to_csv_file( image_real_name + '_' + 'C9'  , getEncodedPixels(C9) )

        save_line_to_csv_file( image_real_name + '_' + 'C10' , getEncodedPixels(C10) )

        save_line_to_csv_file( image_real_name + '_' + 'C11' , getEncodedPixels(C11) )

        save_line_to_csv_file( image_real_name + '_' + 'C12' , getEncodedPixels(C12) )

        save_line_to_csv_file( image_real_name + '_' + 'C13' , getEncodedPixels(C13) )

        save_line_to_csv_file( image_real_name + '_' + 'C14' , getEncodedPixels(C14) )

        save_line_to_csv_file( image_real_name + '_' + 'C15' , getEncodedPixels(C15) )

        save_line_to_csv_file( image_real_name + '_' + 'C16' , getEncodedPixels(C16) )









import time



def getEncodedPixels(image):

    """

    receives a masked image and encodes it to RLE

    :param mask_image:

    :return: string corresponding to the rle of the input image

    """

    pixels = image.flatten()

    # We avoid issues with '1' at the start or end (at the corners of

    # the original image) by setting those pixels to '0' explicitly.

    # We do not expect these to be non-zero for an accurate mask,

    # so this should not harm the score.

    pixels[0] = 0

    pixels[-1] = 0

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2

    runs[1::2] = runs[1::2] - runs[:-1:2]

    return ' '.join(str(x) for x in runs)









def save_line_to_csv_file(name,pixels):

    mylist = [ name , pixels ]

    wr.writerow(mylist)



import csv



with open('trainC1-4.csv', 'w') as myfile:

    wr = csv.writer(myfile, quoting = csv.QUOTE_NONE)

    

    save_line_to_csv_file( 'Image_Label'  , 'EncodedPixels' )

    

    for file_id in range(0, 5546):

        print(file_id)

        image_real_name = train_df.loc[file_id*4,'ImageId']

        print(image_real_name)

        C1,C2,C3,C4, C5,C6,C7,C8,C9,C10, C11,C12,C13,C14, C15,C16 = decode_image_data(image_real_name)

        save_line_to_csv_file( image_real_name + '_' + 'C1'  , getEncodedPixels(C1) )

        save_line_to_csv_file( image_real_name + '_' + 'C2'  , getEncodedPixels(C2) )

        save_line_to_csv_file( image_real_name + '_' + 'C3'  , getEncodedPixels(C3) )

        save_line_to_csv_file( image_real_name + '_' + 'C4'  , getEncodedPixels(C4) )










