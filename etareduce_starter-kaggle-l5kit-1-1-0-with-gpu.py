!pip install --no-index -f /kaggle/input pip==20.2.3
!pip install --no-index -f /kaggle/input l5kit>=1.1.0
import torch

import torchvision

import l5kit

torch.__version__, torchvision.__version__, l5kit.__version__
torch.cuda.is_available()