#########################################
# IMPORTANT: Change the "Internet" settings in the right panel to "Internet Connected"
#########################################

!pip install fastai==0.7.0 --no-deps
# fastai depends also on an older version of torch
!pip install torch==0.4.1 torchvision==0.2.1
# Importing a fastai 0.7.0 package now works!
from fastai.transforms import *