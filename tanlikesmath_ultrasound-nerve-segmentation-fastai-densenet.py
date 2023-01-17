from fastai.vision import *
import pandas as pd

import glob
path_images = Path("../input/train")

path_lbl = path_images
fnames = glob.glob('../input/train/*[!_mask].tif')
lbl_names = glob.glob('../input/train/*_mask.tif')
def get_y_fn(x):

    x = Path(x)

    return path_lbl/f'{x.stem}_mask{x.suffix}'
filter_func = lambda x: str(x) in fnames
size = 224
from fastai.utils.mem import *

#free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

#if free > 8200: bs=8

#else:           bs=4

#print(f"using bs={bs}, have {free}MB of GPU RAM free")

bs=8
class SegLabelListCustom(SegmentationLabelList):

    def open(self, fn): return open_mask(fn, div=True)

class SegItemListCustom(SegmentationItemList):

    _label_cls = SegLabelListCustom

codes = ['0','1']

src = (SegItemListCustom.from_folder(path_images)

       .filter_by_func(filter_func)

       .random_split_by_pct()

       .label_from_func(get_y_fn,classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))

data.path = Path('.')
#data.show_batch(2, figsize=(10,7))
learn = unet_learner(data, models.densenet121, metrics=[dice])
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(10,max_lr = 1e-5)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10,max_lr = slice(1e-6,1e-4))