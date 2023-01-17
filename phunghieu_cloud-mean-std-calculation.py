# DATASET_DIR = '/kaggle/input/understanding-clouds-from-satellite-images-384x576/'

DATASET_DIR = '/kaggle/input/understanding_cloud_organization/'
import numpy as np

import cv2

import os

from tqdm import tqdm_notebook
def compute_sample_mean(im_dir, image_files):

    """

    Parameters:

        im_dir: The directory that contains images.

        image_files: List of image files you need.



    Returns:

        sample-mean

    """



    if len(image_files) > 0:

        image_sum = cv2.imread(os.path.join(im_dir, image_files[0]))

        image_sum = cv2.cvtColor(image_sum, cv2.COLOR_BGR2RGB).astype(np.float64)



        for image_file in tqdm_notebook(image_files[1:]):

            img = cv2.imread(os.path.join(im_dir, image_file))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image_sum += img



        return image_sum/(255*len(image_files))



    return None





def compute_pixel_mean(im_dir, image_files, sample_mean=None):

    """

    Parameters:

        im_dir: The directory that contains images.

        image_files: List of image files you need.

        sample_mean: sample-mean value.



    Returns:

        pixel-mean

    """



    if sample_mean is None:

        sample_mean = compute_sample_mean(im_dir, image_files)



    return np.mean(sample_mean, axis=(0, 1))





def compute_sample_std(im_dir, image_files, sample_mean=None):

    """

    Parameters:

        im_dir: The directory that contains images.

        image_files: List of image files you need.

        sample_mean: sample-mean value.



    Returns:

        sample-std

    """



    if len(image_files) > 0:

        if sample_mean is None:

            sample_mean = compute_sample_mean(im_dir, image_files)



        square_diff_sum = 0

        for image_file in tqdm_notebook(image_files):

            img = cv2.imread(os.path.join(im_dir, image_file))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

            img /= 255.

            square_diff_sum += np.square(np.abs(img - sample_mean))



        return np.sqrt(square_diff_sum/len(image_files))



    return 0





def compute_pixel_std(im_dir, image_files, sample_mean=None, pixel_mean=None):

    """

    Parameters:

        im_dir: The directory that contains images.

        image_files: List of image files you need.

        sample_mean: sample-mean value.

        pixel_mean: pixel-mean value.



    Returns:

        pixel-std

    """



    if len(image_files) > 0:

        if pixel_mean is None:

            if sample_mean is not None:

                pixel_mean = compute_pixel_mean(im_dir, image_files, sample_mean)

            else:

                pixel_mean = compute_pixel_mean(im_dir, image_files)



        square_diff_sum = 0

        for image_file in tqdm_notebook(image_files):

            img = cv2.imread(os.path.join(im_dir, image_file))

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

            img /= 255.

            square_diff_sum += np.sum(np.square(np.abs(img - pixel_mean)), axis=(0, 1))



        return np.sqrt(square_diff_sum/(len(image_files)*img.shape[0]*img.shape[1]))



    return 0
train_images_dir = os.path.join(DATASET_DIR, 'train_images')

image_files = os.listdir(train_images_dir)
sample_mean = compute_sample_mean(train_images_dir, image_files)

sample_std = compute_sample_std(train_images_dir, image_files, sample_mean=sample_mean)

pixel_mean = compute_pixel_mean(train_images_dir, image_files, sample_mean=sample_mean)

pixel_std = compute_pixel_std(train_images_dir, image_files, pixel_mean=pixel_mean)
print('Sample mean:', sample_mean)
print('Sample std:', sample_std)
print('Pixel mean:', pixel_mean)

print('Pixel std:', pixel_std)