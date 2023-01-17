import numpy as np

import imageio

import torch

torch.set_printoptions(edgeitems=2, threshold=50)



%matplotlib inline

import matplotlib.pyplot as plt
dir_path = "/kaggle/input"

vol_arr = imageio.volread(dir_path, 'DICOM')

vol_arr.shape
vol = torch.from_numpy(vol_arr).float()

vol = torch.unsqueeze(vol, 0)



vol.shape
vol
plt.imshow(vol_arr[38]);