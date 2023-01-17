!tar -xzf /kaggle/input/panda-training-dataset/images.gz

!tar -xzf /kaggle/input/panda-training-dataset/masks.gz
!pip install -q fastai2
from fastai2.vision.all import *

from pathlib import Path



path = Path('./')

fnames = get_image_files(path/"images")

def label_func(fn): return path/"masks"/f"{fn.stem}_mask{fn.suffix}"
batch_tfms = [Normalize.from_stats(*imagenet_stats)]



panda = DataBlock(blocks=(ImageBlock, MaskBlock),

                   get_items=get_image_files,

                   splitter=RandomSplitter(),

                   get_y=label_func,

                   batch_tfms=batch_tfms)
dls = panda.dataloaders('images', bs=8)

dls.show_batch(max_n=8)
learn = unet_learner(dls, resnet50,n_out=6)
learn.fine_tune(4)
learn.show_results(max_n=8, figsize=(7,8))