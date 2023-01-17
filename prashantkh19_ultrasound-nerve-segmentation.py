from fastai import *

from fastai.vision import *

import PIL
mask = open_mask("/kaggle/input/ultrasound-nerve-segmentation/train/1_37_mask.tif", div=True)

mask.show()
mask.data; mask.data.max()
class SegmentationLabelListCustom(SegmentationLabelList):

    def open(self, fn): 

        return open_mask(fn, div=True)

class SegmentationItemListCustom(SegmentationItemList):

    _label_cls = SegmentationLabelListCustom
label_func =  lambda fname: fname.with_name(str(fname.stem) + "_mask" + str(fname.suffix))

valid_func =  lambda fname: re.search("1_[0-9]*.tif$", str(fname)) or re.search("2_[0-9]*.tif$", str(fname))

train_filter = lambda fname: "mask" not in str(fname)

src = (SegmentationItemListCustom.from_folder('/kaggle/input/ultrasound-nerve-segmentation/train')

        .filter_by_func(train_filter)

        .split_by_valid_func(valid_func)

        .label_from_func(label_func, classes = ['0', '1']))
data = (src.transform(get_transforms(), size=128, tfm_y=True)

        .databunch(bs=4)

        .normalize(imagenet_stats))

data
data.show_batch(2, figsize=(10, 7))
metrics = [dice]
learn = unet_learner(data, models.resnet34, metrics=metrics)
learn.model_dir = '/kaggle/working/'
lr_find(learn)

learn.recorder.plot()
lr=1e-4

learn.fit_one_cycle(5, lr)
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(5, slice(2e-5, lr/2))
learn.show_results(rows=3, figsize=(9,11))
learn.recorder.plot_losses()
learn.unfreeze()
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-5, 1e-4))
learn.recorder.plot_losses()
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(10, slice(3e-5, 6e-4))
learn.show_results(rows=3, figsize=(9,11))
learn.recorder.plot_metrics()
learn.recorder.plot_losses()
learn.save('/kaggle/working/stage-1')
# data = (src.transform(get_transforms(), size=320, tfm_y=True)

#         .databunch(bs=4)

#         .normalize(imagenet_stats))

# data
# learn_full_res = unet_learner(data, models.resnet34, metrics=metrics)
# learn_full_res.load('/kaggle/working/stage-1')
# learn_full_res.freeze()
# learn_full_res.model_dir = '/kaggle/working/'
# lr_find(learn_full_res)

# learn_full_res.recorder.plot()
# learn_full_res.fit_one_cycle(5, 9e-5)
# learn_full_res.unfreeze()
# lr_find(learn_full_res)

# learn_full_res.recorder.plot()
# learn_full_res.fit_one_cycle(10, slice(5e-6, 5e-5))