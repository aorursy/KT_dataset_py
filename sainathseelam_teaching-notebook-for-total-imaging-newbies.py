import pathlib
import imageio
import numpy as np
import cv2 as cv
import numpy as np
from time import sleep
import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from time import sleep
def msg(message):
    f = open('status.xml','w')
    message = "<msg>"+message+"</msg>"
    f.write(message)
    f.close()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from time import time
from skimage import filters

# Glob the training data and load a single image path
training_paths = pathlib.Path('../input/breast-cancer-icpr-contest').glob('*/*/*.jpg')
training_sorted = sorted([x for x in training_paths])
im_path = training_sorted[46]
im = cv.imread(str(im_path))
img = cv.imread(str(im_path))

# Print the image dimensions
print('Original image shape: {}'.format(im.shape))

def blueRatioHistogram(img):
  t1 = time()
  #img = cv.imread('A00_01.jpg')
  red = img[:,:,2]
  blue = img[:,:,0]
  green = img[:,:,1]

  red = tf.convert_to_tensor(red)
  green = tf.convert_to_tensor(green)
  blue = tf.convert_to_tensor(blue)

  blue = tf.to_float(blue)
  red = tf.to_float(red)
  green = tf.to_float(green)
  #100 * b
  b100 = tf.multiply(blue,100.)

  #r+g
  r_g = tf.add(red,green)

  #r+g+b
  r_g_b = tf.add(r_g,blue)

  one = tf.constant([[1.]])

  #r+g+b+1
  r_g_b_1 = tf.add(r_g_b,one)

  #r+g+1
  r_g_1 = tf.add(r_g,one)

  #factor1 = (100*b)/(r+g+1)
  factor1 = tf.div(b100,r_g_1)

  #256
  t56 = tf.multiply(one,255.)

  #factor2 = 256/(r+g+b+1)
  factor2 = tf.div(t56,r_g_b_1)

  #brh = factor1*factor2
  brh = tf.multiply(factor1,factor2)
  #normalising brh and scaling to 256
  maxb =brh
  a = tf.reduce_max(maxb,[0,1])
  brh = tf.div(brh,a)
  brh = tf.multiply(brh,255.)
  brh = tf.round(brh)
  brh = tf.cast(brh,tf.int32)


  with tf.Session() as sess:
    brh = sess.run(brh)
    #equal = sess.run(equal)
    #maxa = sess.run(a)
  t2 = time()
  t = (t2 - t1)
  #print("Time taken is "+str(t)+"us")
  return brh
brh_img = blueRatioHistogram(im)
cv.imwrite('BRH.jpg',brh_img)
cv.imwrite('img.jpg',img)

# Now, let's plot the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(img)
plt.axis('off')
plt.title('Original Image')

plt.subplot(1,2,2)
plt.imshow(brh_img)
plt.axis('off')
plt.title('BRH image')

plt.tight_layout()
plt.show()

from skimage.filters import threshold_otsu
thresh_val = threshold_otsu(brh_img)
mask = np.where(brh_img > thresh_val, 1, 0)

# Make sure the larger portion of the mask is considered background
if np.sum(mask==0) < np.sum(mask==1):
    mask = np.where(mask, 0, 1)
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
im_pixels = im_gray.flatten()
plt.hist(im_pixels,bins=50)
plt.vlines(thresh_val, 0, 100000, linestyle='--')
plt.ylim([0,50000])
plt.title('Grayscale Histogram')

plt.subplot(1,2,2)
mask_for_display = np.where(mask, mask, np.nan)
plt.imshow(im_gray, cmap='gray')
plt.imshow(mask_for_display, cmap='rainbow', alpha=0.5)
plt.axis('off')
plt.title('Image w/ Mask')

plt.show()
from scipy import ndimage
labels, nlabels = ndimage.label(mask)

label_arrays = []
for label_num in range(1, nlabels+1):
    label_mask = np.where(labels == label_num, 1, 0)
    label_arrays.append(label_mask)

print('There are {} separate components / objects detected.'.format(nlabels))
# Create a random colormap
from matplotlib.colors import ListedColormap
rand_cmap = ListedColormap(np.random.rand(256,3))

labels_for_display = np.where(labels > 0, labels, np.nan)
plt.imshow(im_gray, cmap='gray')
plt.imshow(labels_for_display, cmap=rand_cmap)
plt.axis('off')
plt.title('Labeled Cells ({} Nuclei)'.format(nlabels))
plt.show()
for label_ind, label_coords in enumerate(ndimage.find_objects(labels)):
    cell = im_gray[label_coords]
    
    # Check if the label size is too small
    if np.product(cell.shape) < 10: 
        print('Label {} is too small! Setting to 0.'.format(label_ind))
        mask = np.where(labels==label_ind+1, 0, mask)

# Regenerate the labels
labels, nlabels = ndimage.label(mask)
print('There are now {} separate components / objects detected.'.format(nlabels))
fig, axes = plt.subplots(1,6, figsize=(10,6))

for ii, obj_indices in enumerate(ndimage.find_objects(labels)[0:6]):
    cell = im_gray[obj_indices]
    axes[ii].imshow(cell, cmap='gray')
    axes[ii].axis('off')
    axes[ii].set_title('Label #{}\nSize: {}'.format(ii+1, cell.shape))

plt.tight_layout()
plt.show()
# Get the object indices, and perform a binary opening procedure
two_cell_indices = ndimage.find_objects(labels)[1]
cell_mask = mask[two_cell_indices]
cell_mask_opened = ndimage.binary_opening(cell_mask, iterations=8)
fig, axes = plt.subplots(1,4, figsize=(12,4))

axes[0].imshow(im_gray[two_cell_indices], cmap='gray')
axes[0].set_title('Original object')
axes[1].imshow(mask[two_cell_indices], cmap='gray')
axes[1].set_title('Original mask')
axes[2].imshow(cell_mask_opened, cmap='gray')
axes[2].set_title('Opened mask')
axes[3].imshow(im_gray[two_cell_indices]*cell_mask_opened, cmap='gray')
axes[3].set_title('Opened object')


for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()
def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return " ".join([str(i) for i in run_lengths])

print('RLE Encoding for the current mask is: {}'.format(rle_encoding(label_mask)))
import pandas as pd

def analyze_image(im_path):
    '''
    Take an image_path (pathlib.Path object), preprocess and label it, extract the RLE strings 
    and dump it into a Pandas DataFrame.
    '''
    # Read in data and convert to grayscale
    im_id = im_path.parts[-3]
    im = imageio.imread(str(im_path))
    im_gray = rgb2gray(im)
    
    # Mask out background and extract connected objects
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)    
        labels, nlabels = ndimage.label(mask)
    labels, nlabels = ndimage.label(mask)
    
    # Loop through labels and add each to a DataFrame
    im_df = pd.DataFrame()
    for label_num in range(1, nlabels+1):
        label_mask = np.where(labels == label_num, 1, 0)
        if label_mask.flatten().sum() > 10:
            rle = rle_encoding(label_mask)
            s = pd.Series({'ImageId': im_id, 'EncodedPixels': rle})
            im_df = im_df.append(s, ignore_index=True)
    
    return im_df


def analyze_list_of_images(im_path_list):
    '''
    Takes a list of image paths (pathlib.Path objects), analyzes each,
    and returns a submission-ready DataFrame.'''
    all_df = pd.DataFrame()
    for im_path in im_path_list:
        im_df = analyze_image(im_path)
        all_df = all_df.append(im_df, ignore_index=True)
    
    return all_df
testing = pathlib.Path('../input/stage1_test/').glob('*/images/*.png')
df = analyze_list_of_images(list(testing))
df.to_csv('submission.csv', index=None)
