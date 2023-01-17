# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from glob import glob



import cv2

from skimage.transform import resize

from skimage.io import imread



import os

from tqdm import tqdm

import shutil



from joblib import Parallel, delayed



import numpy as np

from dicom_reader import *

import pydicom



def to_binary(img, lower, upper):

    return (lower <= img) & (img <= upper)



def transform_to_hu(medical_image, image):

    hu_image = image * medical_image.RescaleSlope + medical_image.RescaleIntercept

    hu_image[hu_image < -1024] = -1024

    return hu_image



def window_image(image, window_center, window_width):

    window_image = image.copy()

    image_min = window_center - (window_width / 2)

    image_max = window_center + (window_width / 2)

    window_image[window_image < image_min] = image_min

    window_image[window_image > image_max] = image_max

    return window_image



def resize_normalize(image):

    image = np.array(image, dtype=np.float64)

    image -= np.min(image)

    image /= np.max(image)

    return image



def read_dicom(path, window_widht, window_level):

    image_medical = pydicom.dcmread(path)

    image_data = image_medical.pixel_array



    image_hu = transform_to_hu(image_medical, image_data)

    image_window = window_image(image_hu.copy(), window_level, window_widht)

    image_window_norm = resize_normalize(image_window)



    image_window_norm = np.expand_dims(image_window_norm, axis=2)   # (512, 512, 1)

    image_ths = np.concatenate([image_window_norm, image_window_norm, image_window_norm], axis=2)   # (512, 512, 3)

    return image_ths
def save_train_file(f, out_path, img_size):

    # train image

    name = f.split('/')[-1][:-4]

    img = read_dicom(f, window_widht=400, window_level=0)

    img = resize(img, (img_size, img_size))







    # label image

    label_img = imread('../input/body-morphometry-for-sarcopenia/train/Label/' + name + '.png')

    encode = resize(label_img, (img_size, img_size))*255



    color_im = np.zeros([img_size, img_size, 3])

    for i in range(1,4):

        encode_ = to_binary(encode, i*1.0, i*1.0) * 255

        color_im[:, :, i-1] = encode_

        

    plt.figure(figsize=(8,8))

    plt.subplot(1,2,1)

    plt.grid(False)

    plt.imshow(img)

    plt.subplot(1,2,2)

    plt.grid(False)

    plt.imshow(color_im)

    plt.show()

#     cv2.imwrite('{}/train/{}.png'.format(out_path, name), img)

#     cv2.imwrite('{}/mask/{}.png'.format(out_path, name), encode)





def save_test_file(f, out_path, img_size):

    name = f.split('\\')[-1][:-4]



    img = read_dicom(f, window_widht=1200, window_level=0)

    img = resize(img, (img_size, img_size)) * 255



    cv2.imwrite('{}/test/{}.png'.format(out_path, name), img)



    

def save_train(train_images_names, out_path, img_size=128, n_train=-1, n_threads=1):

    if os.path.isdir(out_path):

        shutil.rmtree(out_path)

    os.makedirs(out_path + '/train', exist_ok=True)

    os.makedirs(out_path + '/mask', exist_ok=True)



    if n_train < 0:

        n_train = len(train_images_names)

    try:

        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_train_file)(

            f, out_path, img_size) for f in tqdm(train_images_names[:n_train]))

    except pydicom.errors.InvalidDicomError:

        print('InvalidDicomError')



        

def save_test(test_images_names, out_path='../dataset128', img_size=128, n_threads=1):

    os.makedirs(out_path + '/test', exist_ok=True)

    try:

        Parallel(n_jobs=n_threads, backend='threading')(delayed(save_test_file)(

            f, out_path, img_size) for f in tqdm(test_images_names))

    except pydicom.errors.InvalidDicomError:

        print('InvalidDicomError')





def main():

    train_fns = sorted(glob('{}/*/*.dcm'.format('../input/body-morphometry-for-sarcopenia/train')))

    test_fns = sorted(glob('{}/*/*.dcm'.format('../input/body-morphometry-for-sarcopenia/test')))

    out_path = '../input/body-morphometry-for-sarcopenia/dataset512'

    img_size = 512

    n_train = 4

    n_threads = 4



    

    save_train_file(train_fns[0], out_path, img_size)

    

#     save_train(train_fns, out_path, img_size, n_train, n_threads)

#     save_test(test_fns, out_path, img_size, n_threads)



if __name__ == '__main__':

    main()