%matplotlib inline

import os



import numpy as np

import openslide

from matplotlib import pyplot as plt
images_dir = "../input/prostate-cancer-grade-assessment/train_images/"

masks_dir = "../input/prostate-cancer-grade-assessment/train_label_masks/"



image_files = os.listdir(images_dir)

mask_files = os.listdir(masks_dir)

mask_files_cleaned = [i.replace("_mask", "") for i in mask_files]

images_with_masks = list(set(image_files).intersection(mask_files_cleaned))

len(image_files), len(mask_files), len(images_with_masks)
def min_max_mask_coordinates(mask, axis=1):

    xy = mask.sum(axis=axis)

    xy = np.nonzero(xy)

    xy_min = np.min(xy)

    xy_max = np.max(xy)

    return xy_min, xy_max



def trim_image_to_mask_size(image, mask):

    x_min, x_max = min_max_mask_coordinates(mask, axis=1)

    y_min, y_max = min_max_mask_coordinates(mask, axis=0)



    image = image[x_min:x_max, y_min:y_max]

    mask = mask[x_min:x_max, y_min:y_max]

    return image, mask





def trim_image_mask_to_min_size(image, mask):

    side = min(image.shape[0], image.shape[1])

    image = image[:side, :side]

    mask = mask[:side, :side]

    return image, mask
rows = 10

cols = 10

offset = 0

plt.figure(figsize=(20, rows*(20/cols)))

plt.subplots_adjust(wspace=0, hspace=0)



for chart, index in enumerate(range(rows*cols), 1):

    image_file = images_with_masks[index]

    mask_file = image_file.replace(".tiff", "_mask.tiff")



    with openslide.OpenSlide(os.path.join(images_dir, image_file)) as image:

        with openslide.OpenSlide(os.path.join(masks_dir, mask_file)) as mask:

            size = image.level_dimensions[-1]

            mask = np.array(mask.get_thumbnail(size=size))[:,:,0]

            image = np.array(image.get_thumbnail(size=size))

            

            for _ in range(2):

                image, mask = trim_image_to_mask_size(image, mask)

                image, mask = trim_image_mask_to_min_size(image, mask)

            image = image * (mask > 0).reshape(mask.shape + (1,))

            image[image == 0 ] = 255

            

            ax = plt.subplot(rows, cols, chart)

            ax.imshow(image)

            ax.set_yticklabels([])

            ax.set_xticklabels([])

            ax.set_xticks([])

            ax.set_yticks([])
rows = 10

cols = 10

offset = 0

plt.figure(figsize=(20, rows*(20/cols)))

plt.subplots_adjust(wspace=0, hspace=0)



for chart, index in enumerate(range(rows*cols), 1):

    image_file = images_with_masks[index]

    mask_file = image_file.replace(".tiff", "_mask.tiff")



    with openslide.OpenSlide(os.path.join(images_dir, image_file)) as image:

        with openslide.OpenSlide(os.path.join(masks_dir, mask_file)) as mask:

            size = image.level_dimensions[-1]

            mask = np.array(mask.get_thumbnail(size=size))[:,:,0]

            image = np.array(image.get_thumbnail(size=size))

            

            for _ in range(2):

                image, mask = trim_image_to_mask_size(image, mask)

                image, mask = trim_image_mask_to_min_size(image, mask)

            image = image * (mask > 1).reshape(mask.shape + (1,))

            image[image == 0 ] = 255

            

            ax = plt.subplot(rows, cols, chart)

            ax.imshow(image)

            ax.set_yticklabels([])

            ax.set_xticklabels([])

            ax.set_xticks([])

            ax.set_yticks([])