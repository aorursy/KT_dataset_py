!pip install ultimate==2.1.2
# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals, absolute_import

from ultimate.mlp import MLP
import numpy as np
import sys, random
from PIL import Image
import matplotlib.pyplot as plt

_DTYPE = np.float64
# im_src = '../input/google.bmp'
im_src = '../input/chess_128.png'
# im_src = '../input/line.png'
# im_src = '../input/man.png'

with Image.open(im_src, mode="r") as _im:
    im = np.asarray(_im.convert("RGB"), dtype=_DTYPE)

plt.imshow(im / 255.0)
plt.axis('off')
plt.show()

im = im / 255.0 - 0.5

print(im.shape, im.max(), im.min())

XSIZE = im.shape[1]
YSIZE = im.shape[0]
SIZE = im.shape[0] * im.shape[1]

train_in = np.zeros((SIZE, 2), dtype=_DTYPE)
train_out = np.zeros((SIZE, 3), dtype=_DTYPE)

for idx in range(SIZE):
    x = idx % XSIZE
    y = idx // XSIZE

    train_in[idx][0] = (x - XSIZE // 2) / (XSIZE // 2) 
    train_in[idx][1] = (y - YSIZE // 2) / (YSIZE // 2)

    train_out[idx][0] = im[y][x][0]
    train_out[idx][1] = im[y][x][1]
    train_out[idx][2] = im[y][x][2]

param = {
    # 3 hidden layers of size 4
    'layer_size': [2,4,4,4,3],
    'loss_type': 'hardmse',
    'regularization': 3,
    'activation': 'a2m2l',
    'leaky': (5 ** 0.5 - 3) / 2,
    'output_range': [-0.5, 0.5],
    'output_shrink': 0.1,
    'rate_init': 0.08, 
    'rate_decay': 0.9, 
    'epoch_train': 10 * 30, 
    'epoch_decay': 10,
    'verbose': 0,
}

mlp = MLP(param).fit(train_in, train_out)

pred = mlp.predict(train_in)

im_out = pred.reshape(im.shape)
im_out = np.clip(im_out + 0.5, a_max=1, a_min=0)

plt.imshow(im_out)
plt.axis('off')
plt.show()
