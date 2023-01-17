# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import cv2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

#Pandas version is 0.23.4, thus we cannot use >=0.24 utility functions such as .to_numpy() etc

print(pd.__version__)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dataframe = pd.read_csv('/kaggle/input/severstal-steel-defect-detection/train.csv')

#Splitting after image id and class id

#print(train_dataframe.head(5))

train_dataframe['Image_ID'] = train_dataframe['ImageId_ClassId'].apply(lambda value: value.split('_')[0])

train_dataframe['Class_ID'] = train_dataframe['ImageId_ClassId'].apply(lambda value: value.split('_')[1])

train_dataframe['hasMask'] = ~ train_dataframe['EncodedPixels'].isna()

print(train_dataframe.head(5))

print(train_dataframe['hasMask'].value_counts())



#Creating the mask dataframe which will be used subsequently for training

mask_count_dataframe = train_dataframe.groupby('Image_ID').agg(np.sum).reset_index()

mask_count_dataframe.sort_values('hasMask', ascending=False, inplace=True)

print(mask_count_dataframe.shape)

mask_count_dataframe.head()

#Fetched from xhlulu

def mask2rle(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



def rle2mask(rle, input_shape):

    width, height = input_shape[:2]

    

    #mask is 1d array of width*height, as we cannot otherwise properly create the ground truth

    mask= np.zeros(width*height).astype(np.uint8)

    

    array = np.asarray([int(x) for x in rle.split()])

    print('Width is: ',width)

    print('Height is: ',height)

    

    #Gets every second element starting at position 0

    starts = array[0::2]

    



    #The additional syntax of a[x::y] means get every yth element starting at position x

    #Thus we get the starting positions and lengths gracefully

    

    #Gets every second element starting at position 1

    lengths = array[1::2]

    

    print('First starting position is: ',starts[0])

    print('First length is: ',lengths[0])



    current_position = 0

    for index, start in enumerate(starts):

        print('Index in starts: {} and start {}'.format(index,start))

        

        #We mark with 1 since we create the ground truth masks in this manner

        mask[int(start):int(start+lengths[index])] = 1

        

        #Advancing in the lengths 

        current_position += lengths[index]

        

    return mask.reshape(height, width).T



def build_masks(rles, input_shape):

    #The depth is the length of rles because we can have a mask for every defect marked in the training set

    depth = len(rles)

    #That is why here we have *input_shape, depth, where depth = number of masks with the dimension of *input_shape

    masks = np.zeros((*input_shape, depth))

   

    #Iterating over all rles(building masks for all defects)

    for i, rle in enumerate(rles):

        if type(rle) is str:

            masks[:, :, i] = rle2mask(rle, input_shape)

    

    return masks



def build_rles(masks):

    width, height, depth = masks.shape

    

    rles = [mask2rle(masks[:, :, i])

            for i in range(depth)]

    

    return rles
#Load the specific RLE from the training dataframe

encoded_rl = train_dataframe[(train_dataframe['Image_ID']=='0002cc93b.jpg') & (train_dataframe['hasMask']==True)]

#As one can notice, the _1 signifies that this picture contains the defect with the label 1

print(encoded_rl)
run_length_encoding = encoded_rl['EncodedPixels'].values

#Run length encoding is a form of lossless data compression

print(type(run_length_encoding))
#Select the sample picture to be read and fetch metadata accordingly

sample_filename = 'db4867ee8.jpg'

sample_image_df = train_dataframe[train_dataframe['Image_ID'] == sample_filename]

print(sample_image_df)



#Get the path for a picture

sample_path = os.path.join("/kaggle/input/severstal-steel-defect-detection/train_images",sample_filename)



#Reading a sample picture

sample_img = cv2.imread(sample_path)



#Retrieve the sample rles, which is an entire dataframe column of length 4 in this case

#The four(4) comes from the fact that we have 4 metadata information about the initial image in the training dataset

#One can see the encoded pixel values below in the comment ===>>> 3 defects, with labels 1, 2, 3 and no defect with label 4 for the given image

"""

43212  db4867ee8.jpg_1  349941 2 350194 6 350447 11 350700 15 350953 1...   

43213  db4867ee8.jpg_2  354411 17 354634 50 354857 82 355096 99 355351...   

43214  db4867ee8.jpg_3                              233729 3008 236801 64   

43215  db4867ee8.jpg_4                                                NaN   

"""

sample_rles = sample_image_df['EncodedPixels'].values

print(sample_rles)

print(len(sample_rles))



sample_masks = build_masks(sample_rles, input_shape=(256, 1600))



#The shape is important as we can see how many masks are generated for an image

print(sample_masks.shape)



#Number of masks for an image

print(sample_masks.shape[2])



fig, axs = plt.subplots(5, figsize=(12, 12))

axs[0].imshow(sample_img)

axs[0].axis('off')



for i in range(sample_masks.shape[2]):

    axs[i+1].imshow(sample_masks[:, :, i])

    axs[i+1].axis('off')



#Last image display is empty because there is no RLE ===>>> no defect with label 4 for that image