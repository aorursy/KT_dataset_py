%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np

import torch

import cv2

import os

from fastai.vision import *
nfolds = 4

MASKS = '../input/siimacr-pneumothorax-segmentation-data-256/masks'

DATA = '../input/hypercolumns-pneumothorax-fastai-0-819-lb'



preds0, items = [], []

for fold in range(nfolds):

    preds0.append(torch.load(os.path.join(DATA,'preds_fold'+str(fold)+'.pt')))

    items.append(np.load(os.path.join(DATA,'items_fold'+str(fold)+'.npy'), 

                         allow_pickle=True))

preds0 = torch.cat(preds0)

pred_sum = preds0.view(preds0.shape[0],-1).int().sum(-1)

items = np.concatenate(items)



ys = []

for item in items:

    img = cv2.imread(str(item).replace('train', 'masks'),0)

    img = torch.Tensor(img).byte().unsqueeze(0)

    img = (img > 127)

    ys.append(img)

ys = torch.cat(ys)

yl = (ys.view(ys.shape[0],-1).int().sum(-1) > 0).byte()
def dice_overall(preds, targs):

    n = preds.shape[0]

    preds = preds.view(n, -1)

    targs = targs.view(n, -1)

    intersect = (preds * targs).float().sum(-1)

    union = (preds+targs).float().sum(-1)

    u0 = union==0

    intersect[u0] = 1

    union[u0] = 2

    return (2. * intersect / union)

    

dices = dice_overall(preds0[yl==1] > 0.22, ys[yl==1])
#write list of negative examples sorted by their dificulty (number of predicted pixels)

#and positive examples sorted by dice

vals_n, sorted_idxs = torch.sort(pred_sum[yl==0], descending=True)

file_ids_n = [p.stem for p in items[to_np(yl==0).astype(bool)][sorted_idxs]]

vals_p, sorted_idxs = torch.sort(dices)

file_ids_p = [p.stem for p in items[to_np(yl==1).astype(bool)][sorted_idxs]]
import json

with open('items_p.txt', 'w') as f:  

    json.dump(file_ids_p, f)

with open('items_n.txt', 'w') as f:  

    json.dump(file_ids_n, f)

    

#how to read the file

with open('items_p.txt', 'r') as f:  

    p = json.load(f)

with open('items_n.txt', 'r') as f:  

    n = json.load(f)
#it looks that selection of 2379 most difficult negative examples makes sense

#for remaining example the number of predicted pixels is less than ~10

import matplotlib.pyplot as plt

print('n_pos = ', len(file_ids_p), ', n_neg = ', len(file_ids_n))

print('n_FP = ', int((vals_n > 255*300.0).sum()))

plt.plot(np.arange(len(vals_n)), vals_n.float()/255)

plt.yscale('log')

plt.xlabel('index')

plt.ylabel('number of pixels')

#threshold used for removal of FP

plt.axhline(300.0, linestyle='--', color='red')

plt.text(4000,400.0,'noise_th') 

plt.show()
print('mean true dice = ', float(vals_p.mean()))

plt.plot(np.arange(len(vals_p)), vals_p)

plt.xlabel('index')

plt.ylabel('dice')

plt.show()