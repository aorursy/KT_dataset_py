# pip installation of torch 1.6

!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
import torch

torch.cuda.is_available()
# installing fastai2 here

!pip install fastai==2.0.13
import torch

torch.cuda.is_available(), torch.__version__
import fastai

fastai.__version__
from fastai.data.all import *

from fastai.vision.all import *
path = Path('../input/men-women-classification')

path.ls()
data_path = path.ls()[0]

data_path.ls()
#function to label images based on their path name

def label_func(fname):

    return fname.parent.stem
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),

                   get_items = get_image_files,

                   get_y     = label_func,

                   splitter  = RandomSplitter(seed = 42),

                   item_tfms = Resize(224),

                   batch_tfms=aug_transforms(max_warp = 0.0))
dls = dblock.dataloaders(data_path)

dls.show_batch(max_n = 9)
#returns the list of all classes

dls.vocab
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.lr_find()
lr = 8e-3

learn.fine_tune(4, lr)
learn.show_results()
interp = Interpretation.from_learner(learn)
interp.plot_top_losses(16, figsize=(15,10))
learn.unfreeze()

learn.lr_find()
lr = 5e-6

learn.fine_tune(4, lr)