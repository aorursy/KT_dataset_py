# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import tensorflow as tf

from sklearn.model_selection import train_test_split                       # used to split dataset

from tensorflow.keras.preprocessing.image import ImageDataGenerator        # used to feed jpg images into model

from tensorflow.keras.applications import Xception                         # load pretrained model

from tensorflow.keras.layers import Dense, Flatten                         # add a normal layer at the end

from matplotlib import pyplot as plt                                       # for data visualization

from skimage.transform import rotate, AffineTransform, warp                # for data augmentation

from skimage.transform import resize                                       # resize image

from skimage import io                                                     # for saving images
train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

train
train_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/train/'

test_dir='/kaggle/input/siim-isic-melanoma-classification/jpeg/test/'
#make a dataframe that has image location

images_df = pd.DataFrame()

images_df['image_address'] = train_dir + train['image_name'] + '.jpg'

images_df['target'] = train['target']

images_df
print('Number of malignant cases = ', images_df['target'].sum())

print('Number of benign cases    = ', images_df['target'].count() - images_df['target'].sum())

print('Ratio of malignant cases   = ', images_df['target'].sum()/images_df['target'].count())
# let's use only ~1200 benign images



#lets get a mask with all the benign cases as true

benign_cases_truth = (images_df['target'] == 0).iloc[:1200]

print('we use only', benign_cases_truth.sum(), 'benign cases')



#apply the mask to get ~1200 benign cases

benign_cases = (images_df[:].iloc[:1200])[:][benign_cases_truth]



# use all the malignant cases:

malignant_cases = images_df[:][images_df['target'] == 1]

print('we use', len(malignant_cases.index), 'malignant cases')



images_df = benign_cases.copy()

images_df = images_df.append(malignant_cases)

# don't worry about the order, train_test_split shuffles by default

images_df
# split the data into training and dev set so we can validate our model

X_train, X_dev, y_train, y_dev = train_test_split(images_df,images_df['target'], test_size=0.2, random_state=1234)
input_shape = (299, 299)

input_shape_with_channels = (299, 299, 3)
os.makedirs('./train/zero')

os.makedirs('./train/one')

os.makedirs('./test/zero')

os.makedirs('./test/one')
i = 0
def augment_and_save(path, label, train_or_test):

    

    if label == 0:

        label = 'zero'

    elif label == 1:

        label = 'one'

    image_name = path[-16:-4]

    save_location = train_or_test+'/'+label+'/'+image_name

    

    #read the image

    image = io.imread(path)

    #resize image

    image_resized = resize(image, input_shape)

    #make rotated image

    rotated = rotate(image_resized, angle=45, mode = 'wrap')

    #flipped

    flipLR = np.fliplr(image_resized)

    flipUD = np.flipud(image_resized)

    

    #save image

    io.imsave(save_location+'_resized.jpg', image_resized)

    io.imsave(save_location+'_rotated.jpg', rotated)

    io.imsave(save_location+'_flipLR.jpg',  flipLR)    

    io.imsave(save_location+'_flipUD.jpg',  flipUD)

    

    global i

    print(i, end = ', ')

    i = i+1
# to make training data

X_train.apply(lambda row : augment_and_save(row['image_address'], 

                                  row['target'], 'train'), axis = 1)



# similarly do the same for dev data

# Note this must not be done for Test data