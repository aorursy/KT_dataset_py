



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/fastai1 dataset/FastAI1 DataSet"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
bs=64
train = "../input/fastai1 dataset/FastAI1 DataSet/train"

valid = "../input/fastai1 dataset/FastAI1 DataSet/valid"

path= "../input/fastai1 dataset/FastAI1 DataSet"

fnames = get_image_files("../input/fastai1 dataset/FastAI1 DataSet/valid/hg")

fnames[:5]
np.random.seed(2)

data = ImageDataBunch.from_folder(path,no_check=True,size=224,bs=bs, num_workers=0)

data.normalize(imagenet_stats)


print(data.classes)

len(data.classes),data.c
data.show_batch(rows=3, figsize=(10,15))
#doc(create_cnn)

learn = cnn_learner(data, models.resnet34, metrics=error_rate,model_dir = '/tmp/')
learn.lr_find()
learn.recorder.plot()

lr=1e-3
learn.fit_one_cycle(5,slice(lr))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(3, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(8,8), dpi=60)
learn.unfreeze()

learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5,slice(1e-6,lr/5))
learn.freeze()
learn.fit_one_cycle(1, max_lr=slice(2e-5,1e-3))
learn.save('stage-1')
learn.load('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
img = open_image(path+'/pie_value.png')

img
classes = ['hg', 'lg', 'pc', 'sg']

data2 = ImageDataBunch.single_from_classes(path, classes, size=224).normalize(imagenet_stats)
learn = cnn_learner(data2, models.resnet34,model_dir = '/tmp/').load('stage-1')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class