!pip install ../input/fastai017-whl/fastai-2.0.13-py3-none-any.whl
!pip install ../input/fastai017-whl/fastcore-1.0.13-py3-none-any.whl
from fastai.vision.all import *
from fastai.text.all import *
from fastai.collab import *
from fastai.tabular.all import *
BLUE = '\033[94m'
BOLD   = '\033[1m'
ITALIC = '\033[3m'
RESET  = '\033[0m'

import fastai; print(BOLD + BLUE + "fastai Version: " + RESET + ITALIC + str(fastai.__version__))
import fastcore; print(BOLD + BLUE + "fastcore Version: " + RESET + ITALIC + str(fastcore.__version__))
import sys; print(BOLD + BLUE + "python Version: " + RESET + ITALIC + str(sys.version))
import torchvision; print(BOLD + BLUE + "torchvision: " + RESET + ITALIC + str(torchvision.__version__))
import torch; print(BOLD + BLUE + "torch version: " + RESET + ITALIC + str(torch.__version__))