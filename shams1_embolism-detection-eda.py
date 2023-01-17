import numpy as np, pandas as pd, pydicom as dcm

import keras

import tensorflow as tf

import matplotlib.pyplot as plt, seaborn as sns

import os, glob

tr = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/train.csv")

te = pd.read_csv("../input/rsna-str-pulmonary-embolism-detection/test.csv")

TRAIN_PATH = "../input/rsna-str-pulmonary-embolism-detection/train/"

files = glob.glob('../input/rsna-str-pulmonary-embolism-detection/train/*/*/*.dcm')

def dicom_to_image(filename):

    im = dcm.dcmread(filename)

    img = im.pixel_array

    img[img == -2000] = 0

    return img
print('Total patients {}'.format(len(os.listdir(TRAIN_PATH))))
plt.axis('off')

plt.imshow(-dcm.dcmread("../input/rsna-str-pulmonary-embolism-detection/train/0003b3d648eb/d2b2960c2bbf/00ac73cfc372.dcm").pixel_array);
tr.head()
f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))

for i in range(20):

    plots[i // 5, i % 5].axis('off')

    plots[i // 5, i % 5].imshow(dicom_to_image(np.random.choice(files[:1000])), cmap=plt.cm.bone) # last 1k images