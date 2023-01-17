!pip install tensorflow==1.12.0
import cv2

import math

import glob

import numpy as np

import os

import tensorflow as tf

import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.cluster import MiniBatchKMeans

from utils import build_empty_kernels

from utils import build_dice_kernels
# function to convert input image to a 2 colors matrix using kMeans algorythm

# it assings 1 to values to 'white' color and -1 value to a black color

def prepare_image_data(filename):

    image = cv2.imread(filename)

    (h, w) = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = image.reshape(-1,1)

    clt = MiniBatchKMeans(n_clusters = 2)

    labels = clt.fit_predict(image)

    quant = clt.cluster_centers_.astype("uint8")[labels]

    quant = quant.reshape((h, w))

    

    quant = np.int32(quant)

    quant[quant == quant.min()] = -1

    quant[quant == quant.max()] = 1

    

    return quant
base_path = '/kaggle/input/d6-dices-images/dataset-images/'

dataset = glob.glob(os.path.join(base_path, '*.jpg'))
# function to plot sample images



def display_samples(data, is_gray=False):

    fig=plt.figure(figsize=(20, 20))

    for i in range(1, 6):

        fig.add_subplot(1, 6, i)

        if is_gray:

            plt.imshow(data[i], cmap='gray', vmin=-1, vmax=1)

        else:

            plt.imshow(data[i])

        plt.axis('off')

    plt.show()
data = [plt.imread(image) for image in dataset[:6]]

display_samples(data)
#displaying processed images:

processed_images = [prepare_image_data(image) for image in dataset[:6]]

display_samples(processed_images, is_gray=True)
#building empty-dice filters

empty_dice_filters = build_empty_kernels()



display_samples(empty_dice_filters, is_gray=True)
# function to compute convolution between a given image and filters



def compute_conv(image_data, kernels):

    image_data = np.expand_dims(image_data, axis=0)

    image_data = np.expand_dims(image_data, axis=3)

    image_data = np.float32(image_data)



    empty_kernels = np.float32(kernels)



    kernels = np.expand_dims(kernels, axis=0)

    kernels = kernels.transpose(2, 3, 0, 1)

    

    res = tf.nn.conv2d(image_data, kernels, [1,1,1,1], padding='SAME')

    res = tf.squeeze(res)

    sess = tf.Session()

    with sess.as_default():

        res = res.eval()

        

    return list(np.int32(res.transpose(2,0,1)))
# function segment original image to images of individual dices based on the result of convolution 

def get_dice_images(image, kernels):

    peaks = []

    dices = []

    size = 17



    original = image.copy()

    

    while True:

        max_val = -999999



        conv_results = compute_conv(image, kernels)

        

        for conv_result in conv_results:

            maxx = conv_result.max()

            

            if maxx > max_val:

                max_val = maxx

                peak = np.where(conv_result==maxx)



        if(max_val < 300):  

            break 

                     

        cx = peak[1][0]

        cy = peak[0][0]

        

        dices.append(original[cy - size:cy + size, cx - size:cx + size].copy())

        peaks.append(peak)

        image = cv2.circle(image, (peak[1][0], peak[0][0]), 17,(0,0,0), -1)

        

    return dices
#crooping original images of individual dices



res = get_dice_images(processed_images[0], empty_dice_filters)

display_samples(res, is_gray=True)
#buidling dice-like kernel filters

dice_kernels, dice_sides = build_dice_kernels()

random_dice_kernels = [dice_kernels[1],dice_kernels[50],dice_kernels[100],dice_kernels[150],dice_kernels[200],dice_kernels[250]]

display_samples(random_dice_kernels, is_gray=True)
#combining everything in one function

def predict(dice_images, dice_sides, dice_kernels):

    labels = []



    for dice_image in dice_images:



        label = 0

        max_val = -999999



        conv_results = compute_conv(dice_image, dice_kernels)

        

        for kernel_id in range(len(conv_results)):

            cur_max = conv_results[kernel_id].max()

            

            if cur_max > max_val:

                max_val = cur_max

                label = dice_sides[kernel_id]



        labels.append(label)

        

    return labels



def process_image(filename, dice_kernels, dice_sides, empty_kernels):

    image = prepare_image_data(filename)

    dice_images = get_dice_images(image, empty_kernels)

    labels = predict(dice_images, dice_sides, dice_kernels)

    return labels
data = [plt.imread(image) for image in dataset[:6]]

labels = [process_image(image, dice_kernels, dice_sides, empty_dice_filters) for image in dataset[:6]]
def display_samples_with_pred_labels(data, labels):

    fig=plt.figure(figsize=(20, 20))

    for i in range(1, 6):

        fig.add_subplot(1, 6, i)

        plt.imshow(data[i])

        plt.axis('off')

        labels[i].sort()

        plt.title(str(labels[i]))

    plt.show()
display_samples_with_pred_labels(data, labels)