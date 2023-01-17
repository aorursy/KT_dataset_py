import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
ship_dir = '../input'
train_image_dir = os.path.join(ship_dir, 'train_v2')
test_image_dir = os.path.join(ship_dir, 'test_v2')
##import gc; gc.enable() # memory is tight
from skimage.util.montage import montage2d as montage
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)

## what does this do??
# ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1#
from skimage.morphology import label
def multi_rle_encode(img):
    labels = label(img[:, :, 0])
    return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]

## file is encoded so you need to use the run-length-encode-and-decode algorithm to get the ships masks 
## mapped onto the images
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
    all_masks = np.zeros((768, 768), dtype = np.int16)
    #if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)
## read in the encoded pixels file, then print the file size.  masks.shape[0] give you the size of the y-axis (number of rows)
## and shape[1] give you the x-axis (number of columns=2)
## masks['ImageId'].value_counts().shape[0] tells you how many unique images there are.  It's counting the unique values
## in the ImageId column
masks = pd.read_csv(os.path.join('../input/',
                                 'train_ship_segmentations_v2.csv'))
print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0]) ## there are duplicate images
masks.head()
# ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1#
## Make sure the encode-decode code is working correctly
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 5))
# get one image's encoded pixels to test
rle_0 = masks.query('ImageId=="00021ddc3.jpg"')['EncodedPixels']
# feed the encoded pixels to function that converts it to image
img_0 = masks_as_image(rle_0)
# show the image
ax1.imshow(img_0[:, :, 0])
ax1.set_title('Image$_0$')
# take the converted image and turn it back to encoded pixels
rle_1 = multi_rle_encode(img_0)
# take the converted encoded pixels and convert to image again
img_1 = masks_as_image(rle_1)
# show the image
ax2.imshow(img_1[:, :, 0])
ax2.set_title('Image$_1$')
# print the length of the 2 image masks to show the encoded and decoded lengths
print('Check Decoding->Encoding',
      'RLE_0:', len(rle_0), '->',
      'RLE_1:', len(rle_1))
## create the test data set, will assign the validation set in model.fit(), already have test data set for after model is trained
# create a column on the data set that says if image has at least one ship or not
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
masks.head()
# only keep records with ships for initial training
masks_wships = masks.dropna()
print(masks_wships.shape[0], 'Num records')
print(masks_wships['ImageId'].value_counts().shape[0]) ## there are duplicate images
masks_wships.head()
## ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1#
BATCH_SIZE=48
IMG_SCALING=(3,3)
def make_image_gen(in_df, batch_size = BATCH_SIZE):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            # c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            c_mask = masks_as_image(c_masks['EncodedPixels'].values)
            if IMG_SCALING is not None:
                c_img = c_img[::IMG_SCALING[0], ::IMG_SCALING[1]]
                c_mask = c_mask[::IMG_SCALING[0], ::IMG_SCALING[1]]
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []
## ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1#
train_gen = make_image_gen(masks_wships)
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())
## ref: https://www.kaggle.com/kmader/baseline-u-net-model-part-1#
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (30, 10))
batch_rgb = montage_rgb(train_x)
batch_seg = montage(train_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg)
ax2.set_title('Segmentations')
ax3.imshow(mark_boundaries(batch_rgb, 
                           batch_seg.astype(int)))
ax3.set_title('Outlined Ships')
fig.savefig('overview.png')
## Build the model
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Dropout
import pickle