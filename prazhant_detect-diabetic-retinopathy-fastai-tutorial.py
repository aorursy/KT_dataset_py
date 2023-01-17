%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
import os

path = '../input/aptos2019-blindness-detection/'

train_csv_path = path +'/train.csv'

train_img_path = path + 'train_images/'

train = pd.read_csv(train_csv_path)



#test path strings

print(train_csv_path)

print(train_img_path)
train.head()
print("There are total {} images in training dataset".format(len(train)))
f_names = get_image_files(train_img_path)

f_names[:5]
il = ImageList.from_folder(train_img_path)

il
il.open(il.items[10])
train['id_code'] = train['id_code'].map(lambda x: (train_img_path + x + '.png'))



#test the paths

print(train['id_code'][1])

print(train['id_code'][2])
tfms = get_transforms(do_flip=True,flip_vert=True,

                      max_rotate=360,max_warp=0,max_zoom=1.1,

                      max_lighting=0.1,p_lighting=0.5)
data = ImageDataBunch.from_df(path = '', df= train, label_col='diagnosis', ds_tfms=tfms,

                              valid_pct=0.2, size=224, bs=64).normalize(imagenet_stats)
data.show_batch(rows = 4, figsize= (12,10))
print(data.classes)
kappa = KappaScore()

kappa.weights = "quadratic"
learn = cnn_learner(data, models.resnet34, metrics=[error_rate, kappa])
learn.model
learn.fit_one_cycle(4)
learn.save('stage-1')
interpret = ClassificationInterpretation.from_learner(learn)

losses,idx = interpret.top_losses()



interpret.plot_top_losses(9, figsize=(15,11))
interpret.plot_confusion_matrix(figsize=(12,12), dpi=60)
interpret.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1')
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-06,1e-03))
data = ImageDataBunch.from_df(path = '', df= train, label_col='diagnosis', ds_tfms=tfms,

                              valid_pct=0.2, size=299, bs=32).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=[error_rate,kappa])
learn.lr_find()

learn.recorder.plot()
# learn.fit_one_cycle(8)
# learn.save('stage-1-50')
# learn.unfreeze()

# learn.fit_one_cycle(3, max_lr=(slice(1e-6,1e-4)))
#If it doesn't, you can always go back to your previous model.

# learn.load('stage-1-50');
# interp = ClassificationInterpretation.from_learner(learn)
# interp.most_confused(min_val=2)