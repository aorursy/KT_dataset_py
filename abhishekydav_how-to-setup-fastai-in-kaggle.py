import sys
!curl -s https://course.fast.ai/setup/colab | bash
!git clone https://github.com/yabhi0807/libml1.git /kaggle/tmp/fastai # This is my repo with all the fastai(updated) libraries 
sys.path.append('/kaggle/tmp/fastai')
!mkdir /kaggle/tmp/data/
!ln -s /kaggle/tmp/fastai /kaggle/working/
!ln -s /kaggle/tmp/data /kaggle/working/
from fastai.transforms import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *
!wget -O /kaggle/tmp/fastai/weights.tgz http://files.fast.ai/models/weights.tgz

!tar xvfz /kaggle/tmp/fastai/weights.tgz -C /kaggle/tmp/fastai
!du -hs /kaggle/working
!du -hs /kaggle/tmp