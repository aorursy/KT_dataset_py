import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

import matplotlib.pyplot as plt

from glob import glob

import seaborn as sns

sns.set_style("whitegrid", {'axes.grid': False})

from scipy import stats

from skimage.io import imread, imshow

from skimage.util import crop

import os

import cv2

from collections import namedtuple

def get_dots_from_image(cropped_dotted, cropped_raw):

    """

    # Get the markers only

    There are also brown markers which are removed by our thresholding and are also not very present in the difference image itself."""

    y_max, x_max, _ = cropped_dotted.shape

    diff = cv2.subtract(cropped_dotted, cropped_raw)

    diff = diff/diff.max()

    diff = cv2.absdiff(cropped_dotted, cropped_raw)

    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)

    ret,th1 = cv2.threshold(gray,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    cnts = cv2.findContours(th1.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    x, y = [], []

    for loc in cnts:

        x.append(loc[0][0][0])

        y.append(loc[0][0][1])

    x = np.array(x)

    y = np.array(y)

    return x,y



labimg=namedtuple('LabeledImage',['image','x','y'])



def load_image_and_labels(img_id):

    """

    Read the images and compute the x,y coordinates of the sea lions

    """

    temp_dotted = cv2.cvtColor(cv2.imread('../input/TrainDotted/{}'.format(img_id)), cv2.COLOR_BGR2RGB)

    temp_raw = cv2.cvtColor(cv2.imread('../input/Train/{}'.format(img_id)), cv2.COLOR_BGR2RGB)

    x,y = get_dots_from_image(temp_dotted,temp_raw)

    return labimg(temp_raw,x,y)
training_image_ids = [os.path.basename(c) for c in glob('../input/Train/*.jpg')]

print(len(training_image_ids),'images, first', training_image_ids[0])
fig, m_axs = plt.subplots(2,2, figsize=(10,10))

for c_ax, t_img_id in zip(m_axs.flatten(), training_image_ids):

    t_img = load_image_and_labels(t_img_id)

    c_ax.imshow(t_img.image)
fig, m_axs = plt.subplots(4,1, figsize=(8,24))

for (ax1), t_img_id in zip(m_axs, training_image_ids):

    t_img = load_image_and_labels(t_img_id)

    ax1.imshow(t_img.image)

    ax1.plot(t_img.x,t_img.y,'r.')
sea_lion_color=[ 196.26725,  182.58025,  168.52425]

mae_sea_lion_map=lambda img: np.abs(img[:,:,0]-sea_lion_color[0])+np.abs(img[:,:,1]-sea_lion_color[1])+np.abs(img[:,:,2]-sea_lion_color[2])
fig, m_axs = plt.subplots(2,2, figsize=(8,12))

for (ax1,ax2), t_img_id in zip(m_axs, training_image_ids):

    t_img = load_image_and_labels(t_img_id)

    ax1.imshow(t_img.image)

    ax1.plot(t_img.x,t_img.y,'r.')

    

    ax2.imshow(mae_sea_lion_map(t_img.image),cmap='bone')

    ax2.plot(t_img.x,t_img.y,'r.')