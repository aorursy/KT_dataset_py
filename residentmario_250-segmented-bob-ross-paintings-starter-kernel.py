import os

os.listdir('../input/segmented-bob-ross-images/train/images/')[:5]
from PIL import Image
imgs_dir = '../input/segmented-bob-ross-images/train/images/'

Image.open(imgs_dir + 'painting53.png')
Image.open(imgs_dir + 'painting224.png')
len(os.listdir('../input/segmented-bob-ross-images/train/images/'))
Image.open(imgs_dir + 'painting336.png')
import numpy as np

segmaps_dir = '../input/segmented-bob-ross-images/train/labels/'

segmap = np.array(Image.open(segmaps_dir + 'painting224.png'))

segmap
np.unique(segmap)
import pandas as pd

pd.read_csv(

    "../input/segmented-bob-ross-images/labels.csv", header=None,

    names=['Class']

)
np.array(Image.open(imgs_dir + 'painting224.png')).shape
np.array(Image.open(imgs_dir + 'painting20.png')).shape
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting224.png')))

axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting224.png')))
fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting20.png')))

axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting20.png')))
fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting10.png')))

axarr[1].imshow(np.array(Image.open(segmaps_dir + 'painting10.png')))
%ls ../input/segmented-bob-ross-images/
import matplotlib.pyplot as plt

fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

axarr[0].imshow(np.array(Image.open(imgs_dir + 'painting74.png')))

axarr[1].imshow(np.array(Image.open('../input/segmented-bob-ross-images/painting74.png')))