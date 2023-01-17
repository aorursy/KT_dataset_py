import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
import skimage.io


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
COMP_DIR = '../input/prostate-cancer-grade-assessment'
train_df = pd.read_csv(os.path.join(COMP_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(COMP_DIR, 'test.csv'))
sub_df = pd.read_csv(os.path.join(COMP_DIR, 'sample_submission.csv'))
model_dir = '../input/panda-public-models'
image_folder = os.path.join(COMP_DIR, 'test_images')
is_test = os.path.exists(image_folder)  # IF test_images is not exists, we will use some train images.

image_folder = image_folder if is_test else os.path.join(COMP_DIR, 'train_images')
df = test_df if is_test else train_df.loc[:100]

class config:
    SZ = 256
    IMG_SIZE = 256
    N = 36
    BS = 8 # batch size
    NUM_WORKERS = 4


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(image_folder)
def get_tiles(img, mode=0):
    result = []
    
    h, w, c = img.shape
    pad_h = (config.SZ - h % config.SZ) % config.SZ + ((config.SZ * mode) // 2)
    pad_w = (config.SZ - w % config.SZ) % config.SZ + ((config.SZ * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], 
                        [pad_w // 2, pad_w - pad_w//2], 
                        [0,0]], constant_values=255)
    img3 = img2.reshape(
        img2.shape[0] // config.SZ,
        config.SZ,
        img2.shape[1] // config.SZ,
        config.SZ,
        3
    )
    
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, config.SZ, config.SZ, 3) # (783, 256, 256, 3)

    # all tile pixel < a piece of white
    n_tiles_with_info = (img3.reshape(img3.shape[0], -1).sum(1) < config.SZ ** 2 * 3 * 255).sum() # number of having image
    
    # supplement
    if len(img3) < config.N:
        img3 = np.pad(img3, [ [0, config.N-len(img3)], [0,0],[0,0],[0,0]], constant_values=255)
        
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:config.N]
    img3 = img3[idxs]
    for i in range(len(img3)):
        result.append({'img':img3[i], 'idx':i})
        
    return result, n_tiles_with_info >= config.N
# sample img
path = os.path.join(image_folder, train_df.loc[2, 'image_id'] + '.tiff')
img = skimage.io.MultiImage(path)[1]
print(img.shape)
# test get_tiles
a, b = get_tiles(img, mode=0)
print(b)
fig, ax = plt.subplots(6, 6, figsize=(10, 10))
for i in range(6):
    for j in range(6):
        ax[i, j].imshow(a[i*6+j]['img'])
def show_grid(path, level, SZ=256, N=36, mode=0, ):
    
    img = skimage.io.MultiImage(path)[level]
    
    h, w, c = img.shape
    pad_h = (SZ - h % SZ) % SZ + ((SZ * mode) // 2)
    pad_w = (SZ - w % SZ) % SZ + ((SZ * mode) // 2)

    img2 = np.pad(img, [[pad_h // 2, pad_h - pad_h // 2], 
                        [pad_w // 2, pad_w - pad_w//2], 
                        [0,0]], constant_values=255)
    print(img2.shape)
    nh, nw, _ = img2.shape
    
    # show padded result
    # to know what mode means
    fig, ax = plt.subplots(1, figsize=(40, 40))
    ax.imshow(img2)
    ax.set_yticks(np.arange(0, nh, SZ))
    ax.set_xticks(np.arange(0, nw, SZ))
    ax.grid(color='black', linestyle='-', linewidth=1)

    # show which tiles are chosen
    img3 = img2.reshape(
        img2.shape[0] // SZ,
        SZ,
        img2.shape[1] // SZ,
        SZ,
        3
    )
    
    new_row, new_col = img3.shape[0], img3.shape[2]
    img3 = img3.transpose(0, 2, 1, 3, 4).reshape(-1, SZ, SZ, 3) # (783, 256, 256, 3)
    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:N]
    for idx in idxs:
        x = (idx % new_col) * SZ
        y = (idx // new_col) * SZ
        rect = patches.Rectangle((x, y), SZ, SZ, linewidth=1, edgecolor='b', facecolor='b', alpha=0.1)
        ax.add_patch(rect)

    plt.show()
show_grid(path=os.path.join(image_folder, train_df.loc[2, 'image_id'] + '.tiff'), 
             level=1, 
             SZ=256, 
             N=36, 
             mode=0)
show_grid(path=os.path.join(image_folder, train_df.loc[2, 'image_id'] + '.tiff'), 
             level=1, 
             SZ=256, 
             N=36, 
             mode=2)
show_grid(path=os.path.join(image_folder, train_df.loc[2, 'image_id'] + '.tiff'), 
             level=2, 
             SZ=128, 
             N=16, 
             mode=0)
show_grid(path=os.path.join(image_folder, train_df.loc[2, 'image_id'] + '.tiff'),
             level=1, 
             SZ=64, 
             N=448, 
             mode=0)