# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import cv2

import matplotlib.pyplot as plt
data_path = '/kaggle/input/understanding_cloud_organization'

train_csv_path = os.path.join('/kaggle/input/understanding_cloud_organization','train.csv')

train_image_path = os.path.join('/kaggle/input/understanding_cloud_organization','train_images')
# load full data and label no mask as -1

train_df = pd.read_csv(train_csv_path).fillna(-1)
# image id and class id are two seperate entities and it makes it easier to split them up in two columns

train_df['ImageId'] = train_df['Image_Label'].apply(lambda x: x.split('_')[0])

train_df['Label'] = train_df['Image_Label'].apply(lambda x: x.split('_')[1])

# lets create a dict with class id and encoded pixels and group all the defaults per image

train_df['Label_EncodedPixels'] = train_df.apply(lambda row: (row['Label'], row['EncodedPixels']), axis = 1)
def rle_to_mask(rle_string, height, width):

    '''

    convert RLE(run length encoding) string to numpy array



    Parameters: 

    rle_string (str): string of rle encoded mask

    height (int): height of the mask

    width (int): width of the mask 



    Returns: 

    numpy.array: numpy array of the mask

    '''

    

    rows, cols = height, width

    

    if rle_string == -1:

        return np.zeros((height, width))

    else:

        rle_numbers = [int(num_string) for num_string in rle_string.split(' ')]

        rle_pairs = np.array(rle_numbers).reshape(-1,2)

        img = np.zeros(rows*cols, dtype=np.uint8)

        for index, length in rle_pairs:

            index -= 1

            img[index:index+length] = 255

        img = img.reshape(cols,rows)

        img = img.T

        return img
train_df.head()
def graph(id_name):

    img = cv2.imread(os.path.join(train_image_path, train_df[train_df['Image_Label']==id_name]['ImageId'].values[0]))

    #CLAHE(OpenCV)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    img1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    #Equalize(OpenCV)

    img_yuv1 = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    img_yuv1[:,:,0] = cv2.equalizeHist(img_yuv1[:,:,0])

    img2 = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)



    mask_decoded = rle_to_mask(train_df[train_df['Image_Label']==id_name]['Label_EncodedPixels'].values[0][1], img.shape[0], img.shape[1])

    fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True, figsize=(20,10))

    ax[0].set_title("{}".format(train_df[train_df['Image_Label']==id_name]['Label'].values[0]), size = 12, color = "blue")

    ax[0].imshow(img);

    ax[1].set_title("CLAHE(OpenCV)", size = 12, color = "red")

    ax[1].imshow(img1);

    ax[2].set_title("Equalize(OpenCV)", size = 12, color = "red")

    ax[2].imshow(img2);

    ax[3].set_title("mask-image", size = 12, color = "red")

    ax[3].imshow(mask_decoded);
for name in train_df[(train_df['EncodedPixels']!=-1)&(train_df['Label']=='Fish')]['Image_Label'].tolist()[:10]:

    graph(name)
for name in train_df[(train_df['EncodedPixels']!=-1)&(train_df['Label']=='Flower')]['Image_Label'].tolist()[:10]:

    graph(name)
for name in train_df[(train_df['EncodedPixels']!=-1)&(train_df['Label']=='Gravel')]['Image_Label'].tolist()[:10]:

    graph(name)
for name in train_df[(train_df['EncodedPixels']!=-1)&(train_df['Label']=='Sugar')]['Image_Label'].tolist()[:10]:

    graph(name)