import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from skimage.io import imread

import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from skimage.util import montage

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

ship_dir = '../input/airbus-ship-detection'

train_image_dir = os.path.join(ship_dir, 'train_v2')

test_image_dir = os.path.join(ship_dir, 'test_v2')

import gc; gc.enable() # memory is tight
data = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations_v2.csv'))



data.head()
from skimage.morphology import label



def multi_rle_encode(img):

    labels = label(img[:, :, 0])

    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]



# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode

def rle_encode(img):

    '''

    img: numpy array, 1 - mask, 0 - background

    Returns run length as string formated

    '''

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)



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



def masks_as_image(in_mask_list):

    # Take the individual ship masks and create a single mask array for all ships

    all_masks = np.zeros((768, 768), dtype = np.uint8)

    for mask in in_mask_list:

        if isinstance(mask, str):

            all_masks |= rle_decode(mask)

    return np.expand_dims(all_masks, -1)
data['ships'] = data['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)

data.dropna(subset=['EncodedPixels'], inplace=True)

unique_img_ids = data.groupby('ImageId').agg({'ships': 'sum'}).reset_index()

unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)



unique_img_ids.head()

data.head()
#unique_img_ids = unique_img_ids.groupby('ships').apply(lambda x : x.sample(10000) if len(x)>10000 else x)

unique_img_ids['ships'].hist(bins=unique_img_ids['ships'].max())

data.drop(['ships'], axis=1, inplace=True)

modified_data = pd.merge(data, unique_img_ids)

modified_data.head(10)
all_images = list(modified_data.groupby('ImageId'))
NUMBER_IN_SAMPLE = 4

np.random.shuffle(all_images)

sample_images = all_images[:NUMBER_IN_SAMPLE]



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))



img = []

seg = []

for i in range(NUMBER_IN_SAMPLE):

    img += [imread(os.path.join(train_image_dir, sample_images[i][0]))]

    seg += [masks_as_image(sample_images[i][1]['EncodedPixels'].values)]



img = np.stack(img, 0)/255.0

seg = np.stack(seg, 0)



print('x', img.shape, img.min(), img.max())

print('y', seg.shape, seg.min(), seg.max())

    

batch_rgb = montage_rgb(img)

batch_seg = montage(seg[:, :, :, 0])

ax1.imshow(batch_rgb)

ax1.set_title('Images')

ax2.imshow(batch_seg)

ax2.set_title('Segmentations')

ax3.imshow(mark_boundaries(batch_rgb, 

                           batch_seg.astype(int)))

ax3.set_title('Outlined Ships')
def mask_to_box(img):

    x_min = img.shape[1]

    y_min = img.shape[0]

    x_max = 0

    y_max = 0

    for i in range(img.shape[0]): # height

        for j in range(img.shape[1]): # width

            if img[i][j] == 1:

                x_min = min(x_min, j)

                y_min = min(y_min, i)

                x_max = max(x_max, j)

                y_max = max(y_max, i)

    return x_min, y_min, x_max, y_max
# too time consuming

#modified_data['bounding_box'] = modified_data['EncodedPixels'].apply(mask_to_box)

#modified_data.head()
def coor_to_box(box_coor, shape=(768,768)):

    img = np.zeros(shape, dtype=np.int16)

    for i in range(box_coor[0],box_coor[2]+1):

        img[box_coor[1]][i] = 1

        img[box_coor[3]][i] = 1

    for j in range(box_coor[1],box_coor[3]+1):

        img[j][box_coor[0]] = 1

        img[j][box_coor[2]] = 1

    return img
NUMBER_IN_SAMPLE = 4

np.random.shuffle(all_images)

sample_images = all_images[:NUMBER_IN_SAMPLE]



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))



img = []

seg = []

box = []

for i in range(NUMBER_IN_SAMPLE):

    img += [imread(os.path.join(train_image_dir, sample_images[i][0]))]

    m_img = masks_as_image(sample_images[i][1]['EncodedPixels'].values)

    seg += [m_img]

    box += [np.expand_dims(coor_to_box(mask_to_box(m_img)),-1)]



img = np.stack(img, 0)/255.0

seg = np.stack(seg, 0)

box = np.stack(box, 0)





print('x', img.shape, img.min(), img.max())

print('y', seg.shape, seg.min(), seg.max())

print('z', box.shape, box.min(), box.max())

    

batch_rgb = montage_rgb(img)

batch_seg = montage(seg[:, :, :, 0])

batch_box = montage(box[:, :, :, 0])

ax1.imshow(batch_rgb)

ax1.set_title('Images')

ax2.imshow(batch_seg)

ax2.set_title('Segmentations')

ax3.imshow(mark_boundaries(batch_rgb, 

                           batch_box.astype(int)))

ax3.set_title('Outlined Ships')