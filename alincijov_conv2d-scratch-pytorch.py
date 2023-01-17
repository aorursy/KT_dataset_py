import torch

import torch.nn as nn

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dt = pd.read_pickle('../input/cifar-100/train')

df = pd.DataFrame.from_dict(dt, orient='index')

df = df.T

df.head()
# Select an image and display

img = np.array(df[df['fine_labels'] == 15].iloc[10]['data']).reshape(3,32,32)

plt.imshow(img.transpose(1,2,0).astype("uint8"), interpolation='nearest')
# convert to pytorch tensor

img = torch.from_numpy(img)
# add the batch_size

img = torch.reshape(img, (1, 3, 32, 32)).type(torch.FloatTensor)
batch_size, channels, h, w = img.shape

print('Batch size:{0}, Channels:{1}, Height:{2}, Width:{3}'.format(batch_size, channels, h, w))
# setup the parameters for Conv2d



kh, kw = 3, 3 # kernel size

dh, dw = 3, 3 # stride
# Create conv

conv = nn.Conv2d(3, 3, (kh, kw), stride=(dh, dw), bias=False)

conv_weight = conv.weight

actual = conv(img)
patches = img.unfold(2, kh, dh).unfold(3, kw, dw)

# batch_size, channels, h_windows, w_windows, kh, kw

print('Patches unfold shape: ', patches.shape)
patches = patches.contiguous().view(batch_size, channels, -1, kh, kw)

# batch_size, channels, windows, kh, kw

print('Patches contiguous shape: ', patches.shape)
nb_windows = patches.size(2)
# Shift the windows into the batch dimension using permute

patches = patches.permute(0, 2, 1, 3, 4)

# batch_size, nb_windows, channels, kh, kw

print('Patches permutation shape', patches.shape)
# Multiply the patches with the weights in order to calculate the conv

result = (patches.unsqueeze(2) * conv_weight.unsqueeze(0).unsqueeze(1)).sum([3, 4, 5])

# batch_size, output_pixels, out_channels

print('After conv operation', result.shape)

result = result.permute(0, 2, 1) # batch_size, out_channels, output_pixels
# assuming h = w

h = w = int(result.size(2)**0.5)

result = result.view(batch_size, -1, h, w)
print('Result shape: ', result.shape, ' Actual shape:', actual.shape)
# Verify the error between actual and result

error = (actual - result).abs().max().item()

print('Max Absolute Error : ', error)