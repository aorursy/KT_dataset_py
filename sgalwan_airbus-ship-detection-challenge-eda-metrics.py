# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from skimage.data import imread

# Input data files are available in the "../input/" directory.


# Globals (!implied parameters to some functions)
train_img_dir = '../input/train/'
train_seg_csv = '../input/train_ship_segmentations.csv'

test_img_dir = '../input/test/'

traincsv = pd.read_csv('../input/train_ship_segmentations.csv')


# Reproducibility: keep it for later!
#np.random.seed(100)


# ::::::::::::::::::::::::::: conventions :::::::::::::::::::::::::::::::::::::::
# encodedpixels: the pixels of the object in the image in run-length encoding
#   a string of 'run-start run-length' whitespace-separtated sequence of integers
#   note: run-start is the pixel position in COLUMN-major order and it is 1-based!
# rle: run-length encoding (sequence)
#   the incarnation of the string 'encodedpixels'
#   in a numpy 2D array (dtype=numpy.uint8) with a row format [run-start, run-length]
#   note: run-start is still 1-based and column-major!
# mask: the full-image binary mask of the object (0->background, 1->object)
#   numpy ndarray (2D) (dtype=numpy.uint8)
# combined_masks: the combined masks of objects as a single mask

def rle_pixels(rle):
    """ returns: the pixel count in the object encoded by 'rle' """
    if rle.size > 0:
        return np.sum(rle[:,1])
    return 0


def rle_dims(rle):
    """ returns: the dimensions (height, width) of the object encoded by 'rle' """
    if rle.size > 0:
        return (np.max(rle[:,1]), len(rle))
    return (0,0)


def encodedpixels2rle(encodedpixels):
    if isinstance(encodedpixels, str):
        return np.array(list(zip(*[iter(int(x) for x in encodedpixels.split())]*2)))
    return np.array([])


def object_dims(encodedpixels):
    """ returns: the dimensions (height, width) of the object encoded by 'encodedpixels' """
    return rle_dims(encodedpixels2rle(encodedpixels))


def object_pixels(encodedpixels):
    """ returns: the number of pixels in the object encoded by 'encodedpixels' """
    return rle_pixels(encodedpixels2rle(encodedpixels))


def rle2mask(rle, shape=(768, 768)):
    """
    rle: 2D numpy array with rows of form [start, run-length]
    shape: (rows, cols) the shape of the referenced image
    """
    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)

    run_ranges = [(start - 1, start + length - 1) for (start, length) in rle]

    for a, b in run_ranges:
        mask[a:b] = 1

    return mask.reshape(shape).T


def mask2rle(mask):
    mask = mask.T.ravel()
    
    start_mask = np.concatenate((mask[0:1] > 0, mask[1:] > mask[0:-1]))
    end_mask = np.concatenate((mask[0:-1] > mask[1:], mask[-1:] > 0))

    run_starts = np.where(start_mask.T.ravel())[0] # 0-based!
    run_lengths = np.where(end_mask.T.ravel())[0] - run_starts + 1

    return np.array(list(zip(run_starts + 1, run_lengths)))


def read_train_image(imgid):
    return(imread(train_img_dir + '/' + imgid))


def read_test_image(imgid):
    return(imread(test_img_dir + '/' + imgid))


def get_train_masks(imgid):
    return [rle2mask(encodedpixels2rle(encodedpixels))
            for encodedpixels in 
                traincsv[traincsv.ImageId == imgid]['EncodedPixels']]


def get_train_combined_masks(imgid):
    return(rle2mask(encodedpixels2rle(' '.join(
        traincsv[traincsv.ImageId == imgid]['EncodedPixels'].fillna('').astype(str)))))


def get_train_objcount(imgid):
    return traincsv[traincsv.ImageId == imgid]['EncodedPixels'].count()
# simple test
print('Mask decode-rencode example:')
print(mask2rle(rle2mask([(1,3), (769+1,766), (1537+1, 10)])))

print('Image 00003e153.jpg has %s object(s)' % get_train_objcount('00003e153.jpg'))
print('Image 6c06acaa5.jpg has %s object(s)' % get_train_objcount('6c06acaa5.jpg'))
# :::::::::::::::::::::::::: metrics ::::::::::::::::::::::::::::::::::::::::
def IoU(mask1, mask2):
    Inter = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    Union = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return Inter / (1e-8 + Union)

def fscore(tp, fn, fp, beta=2.):
    if tp + fn + fp < 1:
        return 1.
    num = (1 + beta ** 2) * tp
    return num / (num + (beta ** 2) * fn + fp)

def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
    predict_masks = [m for m in predict_mask_seq if np.any(m >= 0.5)]
    truth_masks = [m for m in truth_mask_seq if np.any(m >= 0.5)]
    
    if len(truth_masks) == 0:
        tp, fn, fp = 0.0, 0.0, float(len(predict_masks))
        return tp, fn, fp

    pred_hits = np.zeros(len(predict_masks), dtype=np.bool) # 0 miss, 1 hit
    truth_hits = np.zeros(len(truth_masks), dtype=np.bool)  # 0 miss, 1 hit

    for p, pred_mask in enumerate(predict_masks):
        for t, truth_mask in enumerate(truth_masks):
            if IoU(pred_mask, truth_mask) > iou_thresh:
                truth_hits[t] = True
                pred_hits[p] = True

    tp = np.sum(pred_hits)
    fn = len(truth_masks) - np.sum(truth_hits)
    fp = len(predict_masks) - tp

    return tp, fn, fp

def mean_fscore(predict_mask_seq, truth_mask_seq,
              iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7,
                              0.75, 0.8, 0.85, 0.9, 0.95], beta=2.):
    """ calculates the average FScore for the predictions in an image over
    the iou_thresholds sets.
    predict_mask_seq: list of masks of the predicted objects in the image
    truth_mask_seq: list of masks of ground-truth objects in the image
    """
    return np.mean(
        [fscore(tp, fn, fp, beta) for (tp, fn, fp) in 
            [confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh)
                for iou_thresh in iou_thresholds]])
print(traincsv.head())

traincsv.describe()
print("Total inferrences in the train set: " , traincsv.shape[0])

id_images = traincsv.ImageId.unique()
id_images_noships = traincsv[traincsv.EncodedPixels.isna()].ImageId.unique()
id_images_ships = traincsv[traincsv.EncodedPixels.notna()].ImageId.unique()

n_images = id_images.shape[0]
n_images_noships = id_images_noships.shape[0]
n_images_ships = id_images_ships.shape[0]

print("Total no. of images: ", n_images)
print("No. of images with no ships: ", n_images_noships)
print("No. of images with ships: ", n_images_ships)
plt.bar(['No ships', 'Ships'], [n_images_noships, n_images_ships]);
plt.ylabel('Image count');
id_images_obj = traincsv.dropna().groupby('ImageId').count()

id_images_obj.rename({'EncodedPixels': 'ObjCount'}, axis='columns', inplace=True)

objects = id_images_obj.ObjCount.sum()

print("Total No. of object: ", objects)

id_images_obj.describe()
id_images_obj.ObjCount.hist(bins=15)
plt.xlabel('No. of objects');
plt.ylabel('Images');
obj_pixels = traincsv.dropna().EncodedPixels.map(lambda x: object_pixels(x))

obj_pixels.hist()
plt.xlabel('Object size (pixels)');
plt.ylabel('Images');

obj_pixels.describe()
obj_maxdim = traincsv.dropna().EncodedPixels.map(lambda x: max(object_dims(x)))

obj_maxdim.hist()

plt.xlabel('Object length (pixels)');
plt.ylabel('Images');

obj_maxdim.describe()
fig, ax = plt.subplots(4, 4, figsize=(8, 8), dpi=96)

ax = ax.reshape(-1)

for a in ax: 
    a.axis('off')

imgids = np.random.choice(id_images_noships, 8, replace=False)

for i, imgid in enumerate(imgids):
    msk = get_train_combined_masks(imgid)
    ax[2*i].imshow(read_train_image(imgid))
    ax[2*i].set_title(imgid)
    ax[2*i+1].imshow(msk)
fig, ax = plt.subplots(4, 4, figsize=(8, 8), dpi=96)

ax = ax.reshape(-1)

for a in ax: 
    a.axis('off')

imgids = np.random.choice(id_images_ships, 8, replace=False)

for i, imgid in enumerate(imgids):
    msk = get_train_combined_masks(imgid)
    avg_fscore = mean_fscore([msk], get_train_masks(imgid))
    ax[2*i].imshow(read_train_image(imgid))
    ax[2*i].set_title(imgid)
    ax[2*i+1].imshow(msk)
    ax[2*i+1].set_title("%s[%f]" % (get_train_objcount(imgid), avg_fscore))