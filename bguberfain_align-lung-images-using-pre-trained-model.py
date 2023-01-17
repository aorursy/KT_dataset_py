import pydicom

import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

from einops import rearrange, reduce  # pip install einops (amazing lib!)

import cv2

from itertools import starmap



%matplotlib inline
# Configure a few matplotlib parameters

plt.rcParams['figure.figsize'] = [20, 10]

plt.rcParams['image.interpolation'] = 'bilinear'
def extract_image(fname):

    ds = pydicom.read_file(str(fname))

    return ds.pixel_array



fnames = list(Path('../input/siim-acr-pneumothorax-segmentation/sample images/').glob('*.dcm'))

imgs = np.array(list(map(extract_image, fnames)))

imgs.shape
plt.title('Input images')

plt.imshow(rearrange(imgs, '(b1 b2) h w -> (b1 h) (b2 w)', b1=2));
from keras.models import load_model
unet = load_model('../input/u-net-lung-segmentation-montgomery-shenzhen/unet_lung_seg.hdf5', compile=False)
def prepare_input(img, width=512, height=512):

    '''

    Prepare image to be feed into model, according to definitions made by trained model

    '''

    # Resize

    x = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    

    # Normalize

    x = np.float32(x) / 255.

    

    # Add channel axis

    x = x[..., np.newaxis]

    

    return x
X = np.array(list(map(prepare_input, imgs)))

X.shape
y_pred = unet.predict(X)

y_pred.shape
plt.title('Input images with lung segmentation')

plt.imshow(rearrange(X, '(b1 b2) h w () -> (b1 h) (b2 w)', b1=2))

plt.contour(rearrange(y_pred, '(b1 b2) h w () -> (b1 h) (b2 w)', b1=2), levels=[0.5], colors='r');
from scipy import ndimage



# Declare structure used in morphotology opening

morph_structure = np.ones((11, 11))



def crop_segmentation(mask, *others, width=512, height=512, extra_space=0.1):

    '''

    Crop using `mask` as input. `others` are optional arguments that will be croped using `mask`

    as reference.

    '''

    # Binarize mask

    mask_bin = np.squeeze(mask) > 0.5

    

    # Use morphology opening to reduce small structures detected.

    mask_bin = ndimage.morphology.binary_opening(mask_bin, morph_structure)

    

    # This is one of the trickest part: will label each structure and keep only the 3 biggest ones.

    # We assume that these three ones will include the background and two lungs

    mask_bin_label, n_labels = ndimage.label(mask_bin, np.ones((3, 3), dtype=np.uint8))

    used_labels = np.argsort(-np.bincount(mask_bin_label.ravel()))[:3]



    # Remove from mask other objects that are not top-3

    mask_bin &= np.in1d(mask_bin_label.reshape(-1), used_labels).reshape(mask_bin.shape)

    

    # Squeeze horizontal and vertical dimention to find where mask begins and ends

    mask_bin_hor = mask_bin.any(axis=0)

    mask_bin_ver = mask_bin.any(axis=1)



    # Find index of first and last positive pixel

    xmin, xmax = np.argmax(mask_bin_hor), len(mask_bin_hor)-np.argmax(mask_bin_hor[::-1])

    ymin, ymax = np.argmax(mask_bin_ver), len(mask_bin_ver)-np.argmax(mask_bin_ver[::-1])

    

    # Add extra space

    xextra = int((xmax-xmin) * extra_space)

    yextra = int((ymax-ymin) * extra_space)

    xmin -= xextra

    xmax += xextra

    ymin -= yextra

    ymax += yextra

    

    # We will use affine transform to crop image. It will deal with padding image if necessary

    # Note: `pts` will follow a L shape: top left, bottom left and bottom right

    # For details see: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#affine-transformation

    pts1 = np.float32([[xmin, ymin], [xmin, ymax], [xmax, ymax]])

    pts2 = np.float32([[0, 0], [0, height], [width, height]])

    M = cv2.getAffineTransform(pts1, pts2)



    # Crop mask

    mask_crop = cv2.warpAffine(mask, M, (height, width), flags=cv2.INTER_AREA, borderValue=0)

    

    if len(others) > 0:

        # Crop others

        others_crop = tuple(cv2.warpAffine(np.squeeze(other), M, (height, width), flags=cv2.INTER_AREA, borderValue=0) for other in others)

        

        return (mask_crop, ) + others_crop

    else:

        return mask_crop
y_crop, X_crop = map(np.array, zip(*starmap(crop_segmentation, zip(y_pred, X))))
plt.title('Final results')

plt.imshow(rearrange(X_crop, '(b1 b2) h w -> (b1 h) (b2 w)', b1=2))

plt.contour(rearrange(y_crop, '(b1 b2) h w -> (b1 h) (b2 w)', b1=2), levels=[0.5], colors='r');