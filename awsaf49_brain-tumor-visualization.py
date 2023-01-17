import os

import numpy as np

import matplotlib.pyplot as plt

import cv2
integer_to_class = {'1': 'meningioma (1)', '2': 'glioma (2)', '3': 'pituitary tumor (3)'}
from glob import glob

glob('/kaggle/input/*/*/*.npy')
labels = np.load('/kaggle/input/brain-tumor/brain_tumor_dataset/labels.npy')

images = np.load('/kaggle/input/brain-tumor/brain_tumor_dataset/images.npy',allow_pickle=True)

masks = np.load('/kaggle/input/brain-tumor/brain_tumor_dataset/masks.npy',allow_pickle=True)



print(labels.shape)

print(images.shape)

print(masks.shape)
import seaborn as sns

sns.set()

classes, counts = np.unique(labels, return_counts=True)

plt.bar(classes, counts, 

        tick_label=['meningioma (1)', 'glioma (2)', 'pituitary tumor (3)'])



for i, idx in enumerate(classes):

    print('number of {}: {}'.format(integer_to_class[str(idx)], counts[i]))
plt.figure(figsize=(16, 8))

for i, idx in enumerate(np.random.randint(images.shape[0], size=18), start=1):

    plt.subplot(3, 6, i)

    plt.imshow(images[idx], cmap='bone')

    

    # set black pixel as transparent for the mask

    mask = np.ma.masked_where(masks[idx] == False, masks[idx])

    plt.imshow(mask, alpha=0.3, cmap='autumn')

    

    plt.title(integer_to_class[str(labels[idx])])

    plt.axis('off')
def get_bounding_box(mask):

    """

    Return the bounding box of a mask image.

    """

    xmin, ymin, xmax, ymax = 0, 0, 0, 0



    for row in range(mask.shape[0]):

        if mask[row, :].max() != 0:

            ymin = row

            break



    for row in range(mask.shape[0] - 1, -1, -1):

        if mask[row, :].max() != 0:

            ymax = row

            break



    for col in range(mask.shape[1]):

        if mask[:, col].max() != 0:

            xmin = col

            break



    for col in range(mask.shape[1] - 1, -1, -1):

        if mask[:, col].max() != 0:

            xmax = col

            break



    return xmin, ymin, xmax, ymax





def crop_to_bbox(image, bbox, crop_margin=10):

    """

    Crop an image to the bounding by forcing a squared image as output.

    """

    x1, y1, x2, y2 =  bbox

    

    # force a squared image

    max_width_height = np.maximum(y2 - y1, x2 - x1)

    y2 = y1 + max_width_height

    x2 = x1 + max_width_height



    # in case coordinates are out of image boundaries

    y1 = np.maximum(y1 - crop_margin, 0)

    y2 = np.minimum(y2 + crop_margin, image.shape[0])

    x1 = np.maximum(x1 - crop_margin, 0)

    x2 = np.minimum(x2 + crop_margin, image.shape[1])

    

    return image[y1:y2, x1:x2]

from IPython.display import display, clear_output

from tqdm import tqdm



dim_cropped_image = 224



images_cropped = []



for i in tqdm(range(images.shape[0]),leave=True, position = 0):

#     if i % 10 == 0:

#         # print the pourcentage of images processed

#         clear_output(wait=True)

#         display('[{}/{}] images processed: {:.1f} %'

#                 .format(i+1, images.shape[0], (i+1) / images.shape[0] * 100))

        

    bbox = get_bounding_box(masks[i])

    image = crop_to_bbox(images[i], bbox, 20)

    image = cv2.resize(image, dsize=(dim_cropped_image, dim_cropped_image),

                       interpolation=cv2.INTER_CUBIC)

    images_cropped.append(image)

    

# clear_output(wait=True)

# display('[{}/{}] images processed: {:.1f} %'

#         .format(i+1, images.shape[0], (i+1) / images.shape[0] * 100))
images_cropped = np.array(images_cropped)



print(images_cropped.shape)
plt.figure(figsize=(16, 8))

for i, idx in enumerate(np.random.randint(images.shape[0], size=18), start=1):

    plt.subplot(3, 6, i)

    plt.imshow(images_cropped[idx], cmap='bone')

    plt.title(integer_to_class[str(labels[idx])])

    plt.axis('off')