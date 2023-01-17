import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from PIL import Image
import cv2
data = pd.read_csv('/kaggle/input/train_labels.csv')
# 
fig, ax = plt.subplots(1,5, figsize=(20,5))
for i, idx in enumerate(data[data['label'] == 0]['id'][:5]):
    path = os.path.join('/kaggle/input/train/', idx)
    ax[i].imshow(Image.open(path+'.tif'))
    p = Polygon(((32, 32), (64, 32), (64, 64), (32, 64)),
            fc=(0.0, 0.0, 0.0, 0.0), 
            ec=(0.0, 0.9, 0.0 ,0.9), lw=4, linestyle='--')
    ax[i].add_patch(p)
fig, ax = plt.subplots(1,5, figsize=(20,5))
for i, idx in enumerate(data[data['label'] == 1]['id'][:5]):
    path = os.path.join('/kaggle/input/train/', idx)
    ax[i].imshow(Image.open(path+'.tif'))
    p = Polygon(((32, 32), (64, 32), (64, 64), (32, 64)),
            fc=(0.0, 0.0, 0.0, 0.0), 
            ec=(0.9, 0.0, 0.0 ,0.9), lw=4, linestyle='--')
    ax[i].add_patch(p)
