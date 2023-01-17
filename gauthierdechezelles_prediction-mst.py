import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import re
import os
import imageio

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import numpy as np # linear algebra
import random

import tensorflow as tf
for dirname, _, filenames in os.walk('../input/predictions'):
    name_list = []
    for filename in filenames:
        name_list.append(filename)
    name_list.sort()
    predictions = np.array([imageio.imread(os.path.join(dirname, filename)) for filename in name_list[:100]])
    groundtruth = np.array([imageio.imread(os.path.join(dirname, filename)) for filename in name_list[100:]])

print(predictions.shape)
print(groundtruth.shape)

from skimage import measure
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
import skimage.transform as trsf

for ined in range(40,50):
    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(groundtruth[ined])

    fig.add_subplot(1, 4, 2)
    plt.imshow(predictions[ined])


    fel = seg.felzenszwalb(predictions[ined], 1000, sigma=0.80, min_size = 2700)
    fig.add_subplot(1, 4, 3)
    plt.imshow(fel)
    
    i = 0
    while (fel == i).sum() != 0:
        mask = fel == i
        avg = np.mean(predictions[ined][mask])
        avg = 1 if avg > 100 else 0
        fel[mask] = avg
        i += 1
    fig.add_subplot(1, 4, 4)
    plt.imshow(fel)
    
import cv2
for dirname, _, filenames in os.walk('../input/alexandrejeudi'):
    name_list = []
    for filename in filenames:
        name_list.append(filename)
    name_list.sort()
    predictionsFinal = np.array([cv2.resize(cv2.imread(os.path.join(dirname, filename)), dsize=(608,608)) for filename in name_list])
    print(name_list[:10])
print(predictionsFinal.shape)

predictionsFinal = predictionsFinal/255
predictionsFinal = np.array([cv2.GaussianBlur(img, (1,1),0) for img in predictionsFinal])
a = np.array([[0, 1, 2],
              [2, 0, 4],
              [0, 3, 2]])
print(np.array(np.where(a == 0)))
print(np.amin(np.array(np.where(a == 0)), axis=1))
print(np.amax(np.array(np.where(a == 0)), axis=1))
import matplotlib


for ined in range(0,10):
    fig = plt.figure()
    fig.add_subplot(1, 5, 1)
    plt.imshow(predictionsFinal[ined])


    fel = seg.felzenszwalb(predictionsFinal[ined], 5, sigma=1, min_size = 4000)
    fig.add_subplot(1, 5, 2)
    plt.imshow(fel)
    edged = cv2.Canny(np.uint8(fel), 0, 0)
    image = np.zeros(shape=(608,608))
    i = 0
    masks = fel == -1
    while (fel == i).sum() != 0:
        mask = fel == i
        avg = np.mean(predictionsFinal[ined][mask])
        if avg > 100:
            where = np.array(np.where(mask))
            x1, y1 = np.amin(where, axis=1)
            x2, y2 = np.amax(where, axis=1)
            
            if np.count_nonzero(mask == True)/((x2-x1)*(y2-y1)) > 0.3 and (x2-x1)*(y2-y1) > 40*500:
                if (x2-x1)/(y2-y1) < 0.4 or (x2-x1)/(y2-y1) > 1.6:
                    image = cv2.rectangle(image, (y1, x1), (y2, x2), (255, 0, 0) , -1)
                    masks[mask] = True
        i += 1
    fig.add_subplot(1, 5, 3)
    plt.imshow(masks)
    fig.add_subplot(1, 5, 4)
    plt.imshow(image)
    
guesses= []

for ined in range(len(predictionsFinal)):


    fel = seg.felzenszwalb(predictionsFinal[ined], 20000, sigma=0.01, min_size = 20000)
    
    i = 0
    while (fel == i).sum() != 0:
        mask = fel == i
        avg = np.mean(predictionsFinal[ined][mask])
        avg = 1 if avg < 159 else 0
        fel[mask] = avg
        i += 1
    guesses.append(fel)    
print(len(guesses))
guesses = np.array(guesses)*255
for i in range(len(guesses)):
    print(cv2.imwrite("gauthier" + str(i) + ".jpg", guesses[i]))
print(len([el for el in np.array(guesses).flatten() if el == 1])/len(np.array(guesses).flatten()))
foreground_threshold = 0.4 # percentage of pixels > 1 required to assign a foreground label to a patch
from tqdm import tqdm

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 0
    else:
        return 1

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
            
combined = list(zip(guesses, [int(el.split("_")[1].split(".")[0]) for el in name_list]))
masks_to_submission("out.csv", combined)
print("done.")
