import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import h5py
from tqdm import tqdm
import cv2
train_df = pd.read_csv('../input/train.csv')
channels = ['red', 'green', 'blue', 'yellow']
hdf_path = f'./train.hdf5'
def load_image(id):
    img = np.zeros((4, 512, 512), dtype=np.uint8)
    for c, ch in enumerate(channels):
        img[c, ...] = cv2.imread('../input/train/{}_{}.png'.format(id, ch), cv2.IMREAD_GRAYSCALE)
    return img
with h5py.File(hdf_path, mode='w') as train_hdf5:
    train_hdf5.create_dataset("train", (len(train_df), 4, 512, 512), np.uint8)
    for i, id in tqdm(enumerate(train_df['Id'][:100])):    #Remove the [:100] for full dataset
        img = load_image(id)
        train_hdf5['train'][i, ...] = img
randind = np.random.randint(0, len(train_df), 8)
randind = np.sort(randind)
train_hdf5 = h5py.File(hdf_path, "r")
%%timeit
# with h5py.File(hdf_path, "r") as train_hdf5:       # Causes 20% slowdown :(
batch = train_hdf5['train'][randind, ...]
train_hdf5.close()
%%timeit
batch = np.zeros((8, 4, 512, 512), dtype=np.uint8)
for i, ind in enumerate(randind):
    batch[i, ...] = load_image(train_df['Id'][ind])