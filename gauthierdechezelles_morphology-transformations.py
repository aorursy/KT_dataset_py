import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
import os
import imageio

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import numpy as np
import random


from skimage import measure
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import filters
from skimage import io
import skimage.transform as trsf
from skimage.morphology import skeletonize
import math
import cv2
from scipy.ndimage.interpolation import rotate
from tqdm import tqdm




for dirname, _, filenames in os.walk('../input/bestresults/predictions'):
    name_list = []
    for filename in filenames:
        name_list.append(filename)
    name_list.sort()
    predictions = np.array([imageio.imread(os.path.join(dirname, filename)) for filename in name_list])
#Threshold the images
print("hey")
_quantile = np.quantile(predictions.flatten(), 0.70)
print(_quantile)
print(predictions.shape)
predictions = np.array([0 if pixel < _quantile else 1 for pixel in predictions.flatten()]).reshape(94, 608, 608)

results = []

#for each image
for ined in tqdm(range(len(predictions))):
    #plot the image
    fig = plt.figure()
    fig.add_subplot(1, 5, 1)
    plt.imshow(predictions[ined])
    
    #copy the image that we will filter
    fel = predictions[ined].astype('uint8')    
    
    # store the result of each mask
    openings = []
    
    #size of the mask
    size = 60
    #store the superposition of the kernels just for verification
    kernels = np.full((size, size),0, dtype=np.uint8)
    
    #for each line
    for k in range(0, size, 4):
        
        kernel = np.full((size, size),0, dtype=np.uint8)
        kernel[draw.line(k-1, 0, size-1-k, size-1)] = 1
        # apply transformation
        kernels = np.maximum(kernels, kernel)
        opening5 = cv2.morphologyEx(fel, cv2.MORPH_OPEN, kernel, iterations=5,  borderType = cv2.BORDER_CONSTANT, borderValue = 0)
        #flip the mask and repeat
        kernel = kernel.T
        kernels = np.maximum(kernels, kernel)
        opening6 = cv2.morphologyEx(fel, cv2.MORPH_OPEN, kernel, iterations=5,  borderType = cv2.BORDER_CONSTANT, borderValue = 0)
        #add the filtered results to openings
        openings.append(opening5)
        openings.append(opening6)
    
    total = np.maximum.reduce(openings)
        
    #plot the filtered image
    fig.add_subplot(1, 5, 2)
    plt.imshow(total)
    #plot the superposition of the kernels
    fig.add_subplot(1, 5, 3)
    plt.imshow(kernels)
    results.append(total)
    
    
    


#print the pourcentage of pixels considered as a road
print((np.array(res).flatten() == 1).sum()/len(np.array(res).flatten()))

from tqdm import tqdm

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > 0.10:
        return 1
    else:
        return 0

def mask_to_submission_strings(image):
    patch_size = 16
    im = image[0]
    img_number = image[1]
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, guesses):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for image in tqdm(guesses):
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(image))
            
combined = list(zip(res, [int(el.split("_")[1].split(".")[0]) for el in name_list]))
masks_to_submission("out.csv", combined)
print("done.")