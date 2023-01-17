!pip install fastai --upgrade
from fastai.vision.all import *
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

path = Path('../input/covid19-radiography-database/COVID-19 Radiography Database')
files = get_image_files(path)
import PIL
img = PIL.Image.open(files[0])
img
plt.imshow(img, cmap='gray')


import re
def label_func(fname):
    return fname.parent.name
dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   item_tfms = RandomResizedCrop(128, min_scale=0.35), 
                   batch_tfms=Normalize.from_stats(*imagenet_stats))
dls = dblock.dataloaders(path)
dls.show_batch()


dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   batch_tfms=[Rotate(max_deg=45, p=1.),Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path)
dls.show_batch()

dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   batch_tfms=[Brightness(max_lighting=0.2, p=1.),Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path)
dls.show_batch()

dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   batch_tfms=[RandomErasing(p=1., max_count=6, min_aspect=0.5, sl=0.2, sh=0.2),Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path)
dls.show_batch()

dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   batch_tfms=[Zoom(max_zoom=2.1, p=0.5),Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path)
dls.show_batch()

dblock  = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = label_func,
                   splitter  = RandomSplitter(),
                   batch_tfms=[Warp(magnitude=0.2, p=1.0),Normalize.from_stats(*imagenet_stats)])
dls = dblock.dataloaders(path)
dls.show_batch()
