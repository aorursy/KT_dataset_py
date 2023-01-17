import os

from glob import glob

from tqdm import tqdm

import pandas as pd

from skimage.io import imread

import cv2

import numpy as np

import matplotlib.pyplot as plt



plt.style.use('ggplot') 
img_paths = glob(os.path.join('../input/train', '*.jpg'))

gt_dir = '../input/train_masks'



y = []

for i in tqdm(range(len(img_paths[:1000]))):  

    img_path = img_paths[i]    

    gt = imread(os.path.join(gt_dir, os.path.splitext(os.path.basename(img_path))[0]+'_mask.gif'))

    y.append(gt)

    

y = np.array(y)
Xscales = np.array([128/1280, 256/1280, 512/1280, 1024/1280])

Yscales = Xscales/(1918/1280)

mean_dices = []



for xscale, yscale in zip(Xscales, Yscales):

    

    dices = []

    for i in tqdm(range(len(y))):

        

        mask = y[i]

        seg = cv2.resize(mask, dsize=None, fx=xscale, fy=yscale)

        seg = cv2.resize(seg, (1918, 1280))

        

        mask = mask > 127

        seg = seg > 127

        

        dice = 2.0 * np.sum(seg&mask) / (np.sum(seg) + np.sum(mask))

        dices.append(dice)

    

    dices = np.array(dices)

    mean_dices.append(np.mean(dices))



mean_dices = np.array(mean_dices)

        
plt.figure(figsize=(15, 7))

plt.plot(1280*Xscales, mean_dices)

plt.xlabel('pixel_size')

plt.ylabel('dice')

for i in range(len(Xscales)):

    plt.text(1280*Xscales[i], mean_dices[i], '%.5f'%mean_dices[i])

plt.show()