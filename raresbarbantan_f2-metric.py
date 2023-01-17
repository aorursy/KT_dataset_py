# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2
import math
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/train_ship_segmentations.csv')
df.head()
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
def read_image(img_name, type='train'):
    if type=='train':
        path = '../input/train/{}'
    else:
        path = '../input/test/{}'
    img = cv2.imread(path.format(img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_masks(img_name):
    mask_list = df.loc[df['ImageId'] == img_name, 'EncodedPixels'].tolist()
    all_masks = np.zeros((len(mask_list), 768,768))
    for idx, mask in enumerate(mask_list):
        if isinstance(mask, str):
            all_masks[idx] = rle_decode(mask)
    return all_masks

def read_flat_mask(img_name):
    all_masks = read_masks(img_name)
    return np.sum(all_masks, axis=0)


image_with_ships = '00021ddc3.jpg'
image_with_no_ships = '00003e153.jpg'
_, axarr = plt.subplots(1, 2, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[0].imshow(read_image(image_with_ships))
axarr[0].imshow(read_flat_mask(image_with_ships), alpha=0.4)
axarr[1].imshow(read_image(image_with_no_ships))
axarr[1].imshow(read_flat_mask(image_with_no_ships), alpha=0.4)
def iou(img_true, img_pred):
    i = np.sum((img_true*img_pred) >0)
    u = np.sum((img_true + img_pred) >0) + 0.0000000000000000001  # avoid division by zero
    return i/u

m = read_flat_mask(image_with_ships)
print(iou(m, m), iou(0, np.zeros((768, 768))), iou(m, np.ones((768, 768))))

m = read_flat_mask(image_with_no_ships)
print(iou(m, m), iou(m, np.zeros((768, 768))), iou(m, np.ones((768, 768))))
thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

def f2(masks_true, masks_pred):
    # a correct prediction on no ships in image would have F2 of zero (according to formula),
    # but should be rewarded as 1
    if np.sum(masks_true) == np.sum(masks_pred) == 0:
        return 1.0
    
    f2_total = 0
    ious = {}
    for t in thresholds:
        tp,fp,fn = 0,0,0
        for i,mt in enumerate(masks_true):
            found_match = False
            for j,mp in enumerate(masks_pred):
                key = 100 * i + j
                if key in ious.keys():
                    miou = ious[key]
                else:
                    miou = iou(mt, mp)
                    ious[key] = miou  # save for later
                if miou >= t:
                    found_match = True
            if not found_match:
                fn += 1
                
        for j,mp in enumerate(masks_pred):
            found_match = False
            for i, mt in enumerate(masks_true):
                miou = ious[100*i+j]
                if miou >= t:
                    found_match = True
                    break
            if found_match:
                tp += 1
            else:
                fp += 1
        f2 = (5*tp)/(5*tp + 4*fn + fp)
        f2_total += f2
    
    return f2_total/len(thresholds)


m = read_masks(image_with_ships)
print(f2(m, m), f2(m, [np.zeros((768, 768))]), f2(m, [np.ones((768, 768))]))

m = read_masks(image_with_no_ships)
print(f2(m, m), f2(m, [np.zeros((768, 768))]), f2(m, [np.ones((768, 768))]))
subset_images = 2000
random_files = df['ImageId'].unique()
np.random.shuffle(random_files)
#print(random_files[:subset_images])
#print(random_files)
f2_sum = 0
for fname in random_files[:subset_images]:
    mask = read_masks(fname)
    score = f2(mask, [np.zeros((768,768))])
    f2_sum += score
    
print(f2_sum/subset_images)
df['EncodedPixels'].isna().sum() / len(df['ImageId'].unique())