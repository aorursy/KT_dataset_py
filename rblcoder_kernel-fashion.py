import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.image as mpimg

print(os.listdir("../input"))

df_train = pd.read_csv('../input/train.csv')

df_sample_subm = pd.read_csv('../input/sample_submission.csv')
df_sample_subm.head()
df_sample_subm.info()
#https://stackoverflow.com/questions/32603051/pandas-dataframe-how-to-update-multiple-columns-by-applying-a-function

import gc

def updateWidthHeight(row): #row is the value of row. 

    img_path = row.ImageId

    img_path = '../input/test/' + img_path

    img = mpimg.imread(img_path)

    sh = img.shape

    row['Height'] = sh[0]

    row['Width'] = sh[1]

    del img

    gc.collect()

    return row

df_sample_subm=df_sample_subm.apply(updateWidthHeight,axis=1)
df_sample_subm.head()
df_sample_subm.Height.hist()
df_sample_subm.Width.hist()
df_train.head()
import matplotlib.image as mpimg

print(mpimg.imread('../input/train/00000663ed1ff0c4e0132b9b9ac53f6e.jpg').shape)
df_train.info()
encodedPixels = np.fromstring(df_train.iloc[0].EncodedPixels, dtype=int, sep=' ')
encodedPixels.shape
encodedPixels[:16]
#https://www.kaggle.com/kmader/train-simple-xray-cnn

df_train.ClassId.value_counts()[:12]
#https://www.kaggle.com/kmader/train-simple-xray-cnn

df_train.ClassId.value_counts()[-12:]
df_train.ClassId.nunique()
df_train.Height.hist()
df_train.Width.hist()
df_train_value_counts = df_train.ClassId.value_counts().reset_index()

df_train_value_counts.head()
df_train_value_counts.info()
df_train_value_counts.columns=['ClassId', 'Count']

df_train_value_counts.head()
df_sample_subm.head()
df_train.shape
df_sample_subm.shape
df_train.iloc[0]
#https://stackoverflow.com/questions/38390242/sampling-one-record-per-unique-value-pandas-python

df_sample_unique = df_train.groupby('ClassId', group_keys=False).apply(lambda df: df.sample(1,random_state=2019))
df_sample_unique.head()
df_sample_unique.tail()
df_sample_unique_counts = df_sample_unique.merge(df_train_value_counts, on='ClassId')
df_sample_unique_counts.sort_values(by='Count', ascending=False, inplace=True)
df_sample_unique_counts.head()
#https://stackoverflow.com/questions/21104592/json-to-pandas-dataframe

pd.set_option('display.max_columns', 50)

import pprint 

import json

pp = pprint.PrettyPrinter(indent=4)

from pandas.io.json import json_normalize 

with open('../input/label_descriptions.json') as f:

    d = json.load(f)

    print('****categories****')

    df_categories = json_normalize(d['categories'])

    pp.pprint(df_categories)

    print('****attributes****')

    pp.pprint(json_normalize(d['attributes']))

    #pp.pprint(d)
#https://www.kaggle.com/alexcampos4/eda-airbus-ship-detection-challenge

#https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines

#https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_decode(mask_rle, shape):

    '''

    mask_rle: run-length as string formated (start length)

    shape: (height,width) of array to return 

    Returns numpy array, 1 - mask, 0 - background



    '''

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    #print(img.shape)    

    return img.reshape(shape).T

    

    #return img



# def rle_to_pixels(rle_code,height,width):

#     ''' This function decodes Run Length Encoding into pixels '''

#     rle_code = [int(i) for i in rle_code.split()]

    

#     pixels = [(pixel_position % height, pixel_position // width) 

#               for start, length in list(zip(rle_code[0:-1:2], rle_code[1::2])) 

#               for pixel_position in range(start, start + length)]

        

#     return np.asarray(pixels, dtype=int).reshape(height,width)



def apply_mask(image, mask):

    ''' This function saturates the Red and Green RGB colors in the image 

        where the coordinates match the mask'''

    for x, y in mask:

        image[x, y, [0, 1, 2]] = (255, 255, 0)

    return image
#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb

#https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/

#https://towardsdatascience.com/mask-r-cnn-for-ship-detection-segmentation-a1108b5a083

#https://www.kaggle.com/mariammohamed/image-resizing-without-losing-the-thin-edges

%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt

from PIL import Image

# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 44

ncols = 2

#print(mpimg.imread('../input/train/00000663ed1ff0c4e0132b9b9ac53f6e.jpg').shape)

pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols*5, nrows*5)



ctr=0

for index, row in df_sample_unique_counts.head(n=44).iterrows():

     #print(index, row)

          # Set up subplot; subplot indices start at 1

    img_path = row.ImageId

    classId = row.ClassId

    ctr+=1

    sp = plt.subplot(nrows, ncols, ctr,aspect='auto')

    sp.axis('Off') # Don't show axes (or gridlines)

    img_path = '../input/train/' + img_path

    #img = mpimg.imread(img_path)

    img = np.array(Image.open(img_path))

    #encoded_pixels = np.fromstring(row.EncodedPixels, dtype=int, sep=' ')

    #img = apply_mask(img, rle_to_pixels(encoded_pixels))

    #plt.imshow(img)

    #encoded_pixels = np.expand_dims(encoded_pixels, axis=0)

    #decode_p = rle_to_pixels(row.EncodedPixels,row.Height,row.Width)

    #decode_p = rle_decode(row.EncodedPixels,(row.Height,row.Width))

    decode_p = rle_decode(row.EncodedPixels,(row.Width,row.Height))

    plt.imshow(decode_p)

    #plt.title(classId)

    try:

        s_title = df_categories.loc[df_categories.id==int(classId)]['name'].values[0]

    except:

        s_title = classId

    plt.title(s_title)

    

    ctr+=1

    sp = plt.subplot(nrows, ncols, ctr)

    sp.axis('Off') # Don't show axes (or gridlines)

    #img_path = '../input/train/' + img_path

    #img = mpimg.imread(img_path)

    #img = np.array(Image.open(img_path))

    #encoded_pixels = np.fromstring(row.EncodedPixels, dtype=int, sep=' ')

    #img = apply_mask(img, rle_to_pixels(encoded_pixels))

    #plt.imshow(img)

    #encoded_pixels = np.expand_dims(encoded_pixels, axis=0)

    #decode_p = rle_decode(row.EncodedPixels,(row.Height,row.Width))

    plt.imshow(img)

    plt.title(classId)

    

    
# for index, row in df_sample_unique_counts.head(n=24).iterrows():

#      #print(index, row)

#           # Set up subplot; subplot indices start at 1

#     img_path = row.ImageId

#     classId = row.ClassId

#     ctr+=1

#     sp = plt.subplot(nrows, ncols, ctr)

#     sp.axis('Off') # Don't show axes (or gridlines)

#     img_path = '../input/train/' + img_path

#     img = mpimg.imread(img_path)

#     plt.imshow(img)

#     plt.title(classId)    