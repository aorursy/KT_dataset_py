##### INSTALLATION



import sys

sys.path.append('../input/efficientnet-pytorch/EfficientNet-PyTorch/EfficientNet-PyTorch-master')

from efficientnet_pytorch import EfficientNet
##### CHECK PRE-TRAINED WEIGHT FILES



import os

os.listdir('../input/efficientnet-pytorch/')
##### LOAD PRE-TRAINED MODEL



import torch

import torch.nn as nn



model_name = 'efficientnet-b0'

file_name  = '../input/efficientnet-pytorch/efficientnet-b0-08094119.pth'

out_dim    = 1



model = EfficientNet.from_name(model_name)

model.load_state_dict(torch.load(file_name))



model._fc = nn.Linear(model._fc.in_features, out_dim)
##### CHECK



print(model)