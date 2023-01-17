import os

import cv2

import pydicom

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from tqdm.notebook import tqdm
BASE_PATH = '../input/osic-pulmonary-fibrosis-progression-lungs-mask/mask_clear/mask_clear/'
mask = np.zeros((1024, 1024))

count = 0

for k in tqdm(os.listdir(BASE_PATH)):

    ldir = os.listdir(os.path.join(BASE_PATH, k))

    norm = max([int(p[:-4]) for p in os.listdir(os.path.join(BASE_PATH, k))])

    for p in ldir:

        coeff = int(p[:-4]) / norm

        if coeff > 0.2 and coeff < 0.8:

            mask += cv2.resize(cv2.imread(os.path.join(BASE_PATH, k, p), 0), (1024, 1024)) / 255.

            count += 1

mask = mask / count
maskap = cv2.bilateralFilter(mask.astype('float32'),9,25,25)

maskap = (maskap - maskap.min()) / (maskap.max() - maskap.min())
if not os.path.exists('mask'):

    os.mkdir('mask') 
plt.figure(figsize=(10,10))



plt.imshow(maskap)
plt.figure(figsize=(10,10))



plt.imshow(maskap > 0.35);
maska = (maskap > 0.35).astype('uint8') 

maska_r, maska_l = maska.copy(), maska.copy()



maska_r[:, :512] = 0

maska_l[:, 512:] = 0
plt.figure(figsize=(10,10))



plt.imshow(maska_r, alpha=0.5)

plt.imshow(maska_l, alpha=0.5)
cv2.imwrite('mask/left_mask_asymetric.jpg', maska_l)

cv2.imwrite('mask/right_mask_asymetric.jpg', maska_r)
masks = (maskap + maskap[:, ::-1] +

         maskap[::-1, ::-1] + maskap[::-1, :]) / 4.
plt.figure(figsize=(10,10))



plt.imshow(masks);
plt.figure(figsize=(10,10))



plt.imshow(masks > 0.5)
masks = (masks > 0.5).astype(float)
masks_r, masks_l = masks.copy(), masks.copy()



masks_r[:, :512] = 0

masks_l[:, 512:] = 0
plt.figure(figsize=(10,10))



plt.imshow(masks_r, alpha=0.5)

plt.imshow(masks_l, alpha=0.5)
cv2.imwrite('mask/left_mask_simetric.jpg', masks_l)

cv2.imwrite('mask/right_mask_simetric.jpg', masks_r)
import zipfile



def zip_and_remove(path):

    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)

    

    for root, dirs, files in os.walk(path):

        for file in files:

            file_path = os.path.join(root, file)

            ziph.write(file_path)

            os.remove(file_path)

    

    ziph.close()
zip_and_remove('mask') 