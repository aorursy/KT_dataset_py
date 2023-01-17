# ! pip install fastai2
import numpy as np
import torch

from pathlib import Path
from fastai2.vision.all import *
# Set seed for reproducibility
seed = 9
torch.manual_seed(seed)
np.random.seed(seed)
# Although we don't condone it, let's silence warnings for the sake of readability
import warnings
warnings.filterwarnings("ignore")
# Prep DataBlock with item & batch transforms
dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                  get_items=get_image_files,
                  get_y=parent_label,
                  splitter=GrandparentSplitter(),
                  item_tfms=Resize(460),
                  batch_tfms=aug_transforms(size=224, min_scale=0.75))

# Load data
path = Path("../input/food101tiny/")
dls = dblock.dataloaders(path, bs=64)
# Verify number of classes and images
print(dls.train.c, dls.train.n) 
print(dls.valid.c, dls.valid.n)
# Inspect data. Yum! Looks good to me!
dls.show_batch(nrows=1, ncols=6)
# Original plus eight augmentations for brightness/contrast, rotation, and a flip about x=0
dls.show_batch(nrows=2, unique=True)
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.lr_find()
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(7, lr_max=slice(1e-6, 1e-4))
preds, targets = learn.tta()
top_1 = accuracy(preds, targets).item()
top_2 = top_k_accuracy(preds, targets, k=2).item()
print(f"Top-1 Accuracy: {top_1: .4f} | Top-2 Accuracy {top_2: .4f}")
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=3)
learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.lr_find()
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.lr_find()
learn.fit_one_cycle(7, lr_max=slice(1e-6, 1e-4))
def get_dls(path, bs, size):
    """Return a set of train and valid dataloaders with images of a given size."""
    dblock = DataBlock(blocks=(ImageBlock(), CategoryBlock()),
                  get_items=get_image_files,
                  get_y=parent_label,
                  splitter=GrandparentSplitter(),
                  item_tfms=Resize(460),
                  batch_tfms=aug_transforms(size=size, min_scale=0.75))
    return dblock.dataloaders(path, bs=bs)
# First train on images of size 128
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, metrics=accuracy)
learn.fit_one_cycle(3, 3e-3)
# Bump image size to 224
learn.dls = get_dls(path, bs=64, size=224)
learn.fine_tune(4, 1e-3)
# Bump to the max image dim in our dataset
learn.dls = get_dls(path, bs=24, size=512)
learn.fine_tune(6, 1e-3)
# First find a good learning rate
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)
learn.lr_find()
# Train on images of size 128
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)
learn.fit_one_cycle(3, 1e-3)
# Bump image size to 224
learn.dls = get_dls(path, bs=64, size=224)
learn.fine_tune(6, 1e-3)
# Bump to the max image dim in our dataset
learn.dls = get_dls(path, bs=24, size=512)
learn.fine_tune(9, 1e-3)
# Train on images of size 128
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, metrics=accuracy, cbs=MixUp)
learn.fit_one_cycle(5, 3e-3)
# Bump image size to 224
learn.dls = get_dls(path, bs=64, size=224)
learn.fine_tune(8, 1e-3)
# Bump to the max image dim in our dataset
learn.dls = get_dls(path, bs=24, size=512)
learn.fine_tune(12, 1e-3)
# First find a good learning rate
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy, cbs=MixUp)
learn.lr_find()
# Train on images of size 128
dls = get_dls(path, bs=128, size=128)
learn = cnn_learner(dls, resnet50, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy, cbs=MixUp)
learn.fit_one_cycle(4, 1e-3)
# Bump image size to 224
learn.dls = get_dls(path, bs=64, size=224)
learn.fine_tune(8, 1e-3, freeze_epochs=2)
# Bump to the max image dim in our dataset
learn.dls = get_dls(path, bs=24, size=512)
learn.fine_tune(16, 1e-3, freeze_epochs=2)