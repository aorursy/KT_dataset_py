import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *
import matplotlib.pyplot as plt
path = Path('/kaggle/input/cell-images-for-detecting-malaria/cell_images')
path
path.ls()
k = path/'cell_images'
k.ls()
imglist = ImageList.from_folder(k)
print(len(imglist.items))

print(imglist[125].shape)


print(imglist[120].shape)
np.random.seed(20)

src = imglist.split_by_rand_pct(0.2).label_from_folder()
tfms = get_transforms(flip_vert= True, max_warp= 0)
data = (src.transform(tfms, size = 128).databunch(bs = 16).normalize(imagenet_stats))
data.show_batch(rows = 3, figsize= (15,11))
print(f"""names of the classes in the data are: {data.classes}\n
        Size of the training set is: {data.train_ds}\n
        Size of the validation set is: {data.valid_ds}\n
        """)
learn = create_cnn(data, models.resnet34, metrics = accuracy, model_dir = '/kaggle/working/')
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, slice(1e-3))
learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (7,7), dpi = 60)
interp.plot_top_losses(9, figsize = (10,10))
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr= slice(1e-5, 1e-4))
learn.save('stage -2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize = (8,8), dpi = 60)
interp.plot_top_losses(9, figsize = (10,10))
learn.show_results()
