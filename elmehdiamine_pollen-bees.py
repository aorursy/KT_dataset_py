%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input/pollendataset/PollenDataset"))
images = get_image_files("../input/pollendataset/PollenDataset/images")
def get_labels(f_path): return 'N' if 'N' in str(f_path) else 'P'
f_path = "../input/pollendataset/PollenDataset/images"

bs = 64

tfms = get_transforms(flip_vert=False, max_zoom=1.)

data = ImageDataBunch.from_name_func(f_path, images, label_func=get_labels, ds_tfms=tfms, size=224, bs=bs

                                    ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(10,9))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir='/kaggle/working')
learn.lr_find()

learn.recorder.plot()
lr = 0.01
learn.fit_one_cycle(7, slice(lr))
learn.recorder.plot_losses()
learn.save('stage-1-rn50')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4, 0.0007))
learn.recorder.plot_losses()
learn.save('stage-2-rn50')
data2 = ImageDataBunch.from_name_func(f_path, images, label_func=get_labels, ds_tfms=tfms, size=299, bs=bs

                                    ).normalize(imagenet_stats)
learn.data = data2
learn.freeze()
learn.lr_find()

learn.recorder.plot()
lr = 1e-3/2
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1-299-rn50')
learn.unfreeze()
learn.fit_one_cycle(5, slice(1e-5, lr/5))
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(2, figsize=(15,11), heatmap=False)