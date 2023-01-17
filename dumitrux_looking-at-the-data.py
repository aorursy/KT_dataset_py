# ensure that any edits to libraries you make are reloaded here automatically
%reload_ext autoreload
%autoreload 2

# ensure that any charts or images displayed are shown in this notebook
%matplotlib inline
import os
# fastai V1 library which sits on top of Pytorch 1.0
from fastai.vision import *
# to avoid warning of PyTorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
path = '../input/architectural-styles-dataset/architectural-styles-dataset'
np.random.seed(42)
tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=10, max_zoom=1.1, 
                      max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=tfms, size=224, num_workers=4, padding_mode='reflection', bs=64).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(9,9))
