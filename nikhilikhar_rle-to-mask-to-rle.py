import cv2

import pandas as pd

import numpy as np

from itertools import groupby

from matplotlib import pyplot as plt

category_num = 4+1

train_df = pd.read_csv('../input/train.csv')

train_df[['ImageId', 'ClassId']] = train_df['ImageId_ClassId'].str.split('_', expand=True)

# ClassId subtracted by 1

train_df['ClassId'] = train_df['ClassId'].astype(int) - 1  

train_df.head()

def make_mask_img(segment_df):

    seg_width = 1600

    seg_height = 256

    seg_img = np.full(seg_width*seg_height, category_num-1, dtype=np.int32)

    for encoded_pixels, class_id in zip(segment_df["EncodedPixels"].values, segment_df["ClassId"].values):

        if pd.isna(encoded_pixels): continue

        pixel_list = list(map(int, encoded_pixels.split(" ")))

        for i in range(0, len(pixel_list), 2):

            start_index = pixel_list[i] -1 

            index_len = pixel_list[i+1] 

            seg_img[start_index:start_index+index_len] = int(class_id) 

    seg_img = seg_img.reshape((seg_height, seg_width), order='F')

    return seg_img



def encode(input_string):

    return [(len(list(g)), k) for k,g in groupby(input_string)]



def run_length(label_vec):

    encode_list = encode(label_vec)

    index = 1

    class_dict = {}

    for i in encode_list:

        if i[1] != category_num-1:

            if i[1] not in class_dict.keys():

                class_dict[i[1]] = []

            class_dict[i[1]] = class_dict[i[1]] + [index, i[0]]

        index += i[0]

    return class_dict
name = '0025bde0c.jpg'

df = train_df[train_df['ImageId']==name]

df.head()
mask = make_mask_img(df)

plt.imshow(mask)
plt.imshow(cv2.imread('../input/train_images/' + name))
d = run_length(mask.T.ravel())

nmask = {}

nmask['EncodedPixels'] = []

nmask['ClassId'] = []

for k,v in d.items():

    nmask['ClassId'].append(str(k))

    nmask['EncodedPixels'].append(' '.join(map(str,v)))

for i in range(4):

    if str(i) not in nmask['ClassId']:

        nmask['ClassId'].append(str(i))

        nmask['EncodedPixels'].append(np.nan)

nmask = pd.DataFrame.from_dict(nmask)

nmask
del df['ImageId_ClassId']

del df['ImageId']

df = df.reset_index(drop=True)

df
np.array_equal(df, nmask), df.equals(nmask.reset_index(drop=True))
df['EncodedPixels'].iloc[2] == nmask['EncodedPixels'].iloc[0], df['EncodedPixels'].iloc[3] == nmask['EncodedPixels'].iloc[1]
plt.imshow(make_mask_img(nmask))