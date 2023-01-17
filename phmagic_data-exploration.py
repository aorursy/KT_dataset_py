# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2 as cv
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape=(768, 768)):
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
    return img.reshape(shape).T  # Needed to align to RLE direction

def get_mask(img_id, df):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    masks = df.loc[img_id]['EncodedPixels']
    if(type(masks) == float): return img.reshape(shape)
    if(type(masks) == str): masks = [masks]
    for mask in masks:
        s = mask.split()
        for i in range(len(s)//2):
            start = int(s[2*i]) - 1
            length = int(s[2*i+1])
            img[start:start+length] = 1
    return img.reshape(shape).T
def get_ship_size(px):
    shape = (768,768)
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    s = px.split()
    for i in range(len(s)//2):
        start = int(s[2*i]) - 1
        length = int(s[2*i+1])
        img[start:start+length] = 1
    return img.reshape(shape).T.sum()
train_df = pd.read_csv('../input/train_ship_segmentations_v2.csv')
train_df.set_index('ImageId', inplace=True)
train_df.head()
no_ships_df = train_df[train_df['EncodedPixels'].isna()]
print(len(no_ships_df))
# Sample image with ships
ships_df = train_df[~train_df['EncodedPixels'].isna()]
print(len(ships_df))
ships_df.head()
ship_count = ships_df.groupby([ships_df.index]).size()
plt.hist(ship_count, bins=range(15))
plt.xticks(range(20))
plt.title('Number of ships per image')
plt.ylabel('Number of images in train set')
plt.show()
ships_df['size'] = ships_df.loc[:, 'EncodedPixels'].apply(get_ship_size)
ships_df.head()
print('Max ship size',np.max(ships_df['size']))
print('Min ship size', np.min(ships_df['size']))

plt.hist(ships_df['size'], bins=[1,10,100,1000,10000,25000])
# plt.xticks(range(20))
plt.title('Ship size (pixels) per image')
plt.ylabel('Number of ships in train set')
plt.xlabel('Ship size (pixels)')
plt.show()
def plot_ship_masks(sample_imgids):
    fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
    fig.set_size_inches(20, 10)
    fig.tight_layout()
    for i, imgid in enumerate(sample_imgids):
        col = i % 5
        row = i // 5

        img = cv.imread('../input/train_v2/{}'.format(imgid))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        mask = get_mask(imgid, ships_df)
        plot = ax[row, col]
        plot.set_title(imgid)
        plot.axis('off')
        plot.imshow(img)
        
        plot = ax[row+1, col]
        plot.axis('off')
        plot.imshow(mask)
        
# sample_imgids = list(set(ships_df.index))[:5]

sample_imgids = list(ship_count[ship_count < 2].index)[:5]
plot_ship_masks(sample_imgids)

sample_imgids = list(ship_count[(ship_count > 1) & (ship_count < 6)].index)[:5]
plot_ship_masks(sample_imgids)

sample_imgids = list(ship_count[(ship_count > 6)].index)[:5]
plot_ship_masks(sample_imgids)

img_id = '01541263e.jpg'
img = cv.imread('../input/train_v2/{}'.format(img_id))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.figure(figsize=(20,20))
plt.imshow(img)

plt.axis('off')
plt.show()
# Empty images
sample = train_df[train_df['EncodedPixels'].isna()].sample(10)

fig, ax = plt.subplots(2, 5, sharex='col', sharey='row')
fig.set_size_inches(20, 10)
fig.tight_layout()
for i, imgid in enumerate(sample.index):
    col = i % 5
    row = i // 5
    
    img = cv.imread('../input/train_v2/{}'.format(imgid))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plot = ax[row, col]
    plot.axis('off')
    plot.imshow(img)
    
# sample_imgids = list(set(ships_df.index))[:5]

sample_imgids = list(ships_df[ships_df['size'] > 23000].index)[:5]
plot_ship_masks(sample_imgids)


