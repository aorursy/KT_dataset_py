import os

import math

import random



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import cv2
def set_seed(seed):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    

    

SEED = 42

set_seed(SEED)
BASE_PATH = '../input/gan-getting-started/'

MONET_PATH = os.path.join(BASE_PATH, 'monet_jpg')

PHOTO_PATH = os.path.join(BASE_PATH, 'photo_jpg')
def print_folder_statistics(path):

    d_image_sizes = {}

    for image_name in os.listdir(path):

        image = cv2.imread(os.path.join(path, image_name))

        d_image_sizes[image.shape] = d_image_sizes.get(image.shape, 0) + 1

        

    for size, count in d_image_sizes.items():

        print(f'shape: {size}\tcount: {count}')





print(f'Monet images:')

print_folder_statistics(MONET_PATH)

print('-' * 10)

print(f'Photo images:')

print_folder_statistics(PHOTO_PATH)

print('-' * 10)
def batch_visualization(path, n_images, is_random=True, figsize=(16, 16)):

    plt.figure(figsize=figsize)

    

    w = int(n_images ** .5)

    h = math.ceil(n_images / w)

    

    all_names = os.listdir(path)

    

    image_names = all_names[:n_images]

    if is_random:

        image_names = random.sample(all_names, n_images)

    

    for ind, image_name in enumerate(image_names):

        img = cv2.imread(os.path.join(path, image_name))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

        plt.subplot(h, w, ind + 1)

        plt.imshow(img)

        plt.axis('off')

    

    plt.show()
batch_visualization(MONET_PATH, 1, is_random=True, figsize=(5, 5))
batch_visualization(MONET_PATH, 4, is_random=True, figsize=(10, 10))
batch_visualization(MONET_PATH, 9, is_random=True)
batch_visualization(MONET_PATH, 16, is_random=True)
batch_visualization(MONET_PATH, 300, is_random=False)
batch_visualization(PHOTO_PATH, 16, is_random=True)
def color_hist_visualization(image_path, figsize=(16, 4)):

    plt.figure(figsize=figsize)

    

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    plt.subplot(1, 4, 1)

    plt.imshow(img)

    plt.axis('off')

    

    colors = ['red', 'green', 'blue']

    for i in range(len(colors)):

        plt.subplot(1, 4, i + 2)

        plt.hist(

            img[:, :, i].reshape(-1),

            bins=25,

            alpha=0.5,

            color=colors[i],

            density=True

        )

        plt.xlim(0, 255)

        plt.xticks([])

        plt.yticks([])

    

    

    plt.show()
img_path = '../input/gan-getting-started/monet_jpg/000c1e3bff.jpg'

color_hist_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/05144e306f.jpg'

color_hist_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/16dabe418c.jpg'

color_hist_visualization(img_path)
def channels_visualization(image_path, figsize=(16, 4)):

    plt.figure(figsize=figsize)

    

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    plt.subplot(1, 4, 1)

    plt.imshow(img)

    plt.axis('off')

    

    for i in range(3):

        plt.subplot(1, 4, i + 2)

        tmp_img = np.full_like(img, 0)

        tmp_img[:, :, i] = img[:, :, i]

        plt.imshow(tmp_img)

        plt.xlim(0, 255)

        plt.xticks([])

        plt.yticks([])

    

    

    plt.show()
img_path = '../input/gan-getting-started/monet_jpg/51db3fc011.jpg'

color_hist_visualization(img_path)

channels_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/1814cc6632.jpg'

color_hist_visualization(img_path)

channels_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/4995c04b1a.jpg'

color_hist_visualization(img_path)

channels_visualization(img_path)
def grayscale_visualization(image_path, figsize=(8, 4)):

    plt.figure(figsize=figsize)

    

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

    plt.subplot(1, 2, 1)

    plt.imshow(img)

    plt.axis('off')

    

    plt.subplot(1, 2, 2)

    tmp_img = np.full_like(img, 0)

    for i in range(3):

        tmp_img[:, :, i] = img.mean(axis=-1)

    plt.imshow(tmp_img)

    plt.axis('off')

    

    

    plt.show()
img_path = '../input/gan-getting-started/monet_jpg/5c79cfe0b3.jpg'

grayscale_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/990ed28f62.jpg'

grayscale_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/fd63a333f1.jpg'

grayscale_visualization(img_path)



img_path = '../input/gan-getting-started/monet_jpg/bf6db09354.jpg'

grayscale_visualization(img_path)