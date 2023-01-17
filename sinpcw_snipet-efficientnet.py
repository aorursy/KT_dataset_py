import sys



sys.path.append('/kaggle/input/public-models/EfficientNet-PyTorch-master')
import torch

import torch.nn as nn



from efficientnet_pytorch import EfficientNet
# write your inference code

# (for example)

model = EfficientNet.from_name('efficientnet-b0')

model