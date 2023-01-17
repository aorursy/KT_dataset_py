%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
bs=16
from pathlib import Path
path = Path("../input/earphones/earphone_dataset")
path.ls()
mi = path/'redmi_airdots'
galaxy = path/'galaxy_buds'
airpods = path/'iphone_airpods'
# fn_mi = get_image_files(mi)
# fn_galaxy = get_image_files(galaxy)
# fn_airpods = get_image_files(airpods)
mi.ls()
fn_paths = []
fn_paths = fn_paths + mi.ls() + galaxy.ls() + airpods.ls()
fn_paths
# import re
tfms = get_transforms(do_flip=False)
pat = r"/([^/]*)/[^/]*.jpg$"
data = ImageDataBunch.from_name_re(path, fn_paths, pat=pat, ds_tfms=tfms, size=224, bs=bs)
data.classes
# labels = [('mi' if '/redmi_airdots/' in str(x) else 'airpods') for x in fn_paths]
# labels[:5]
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)
len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)