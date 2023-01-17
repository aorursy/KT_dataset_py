import numpy as np

import torch

import os

import imageio

import requests



from PIL import Image

from io import BytesIO
url = "https://raw.githubusercontent.com/deep-learning-with-pytorch/dlwpt-code/master/data/p1ch2/bobby.jpg"



response = requests.get(url)



img = Image.open(BytesIO(response.content))

img
img_arr = imageio.imread(response.content)

img_arr.shape
img = torch.from_numpy(img_arr)

out = img.permute(2, 0, 1)
img
out
batch_size = 3

batch = torch.zeros(batch_size, 3, 256, 256, dtype=torch.uint8)
data_dir = '../input/'

filenames = [name for name in os.listdir(data_dir) if os.path.splitext(name)[-1] == '.png']



for i, filename in enumerate(filenames):

    img_arr = imageio.imread(os.path.join(data_dir, filename))

    img_t = torch.from_numpy(img_arr)

    img_t = img_t.permute(2, 0, 1)     # Here we keep only the first three channels.

    img_t = img_t[:3]                  # Sometimes images also have an alpha channel

    batch[i] = img_t                   # indicating transparency, but our network only wants RGB input.

    

batch
batch.shape
batch = batch.float()

batch /= 255.0

batch
n_channels = batch.shape[1]

for c in range(n_channels):

    mean = torch.mean(batch[:, c])

    std = torch.std(batch[:, c])

    batch[:, c] = (batch[:, c] - mean) / std
batch