!pip install --upgrade pip
!pip install -U torch_nightly -f https://download.pytorch.org/whl/nightly/cu90/torch_nightly.html
import torch
torch.__version__
import fastai
fastai.__version__
