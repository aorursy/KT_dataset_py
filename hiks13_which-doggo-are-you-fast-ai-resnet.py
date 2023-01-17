# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from fastai.vision import *

from fastai.metrics import error_rate,accuracy







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import gc

gc.collect()
size = 224

bs = 64

path = "../input/"

np.random.seed(1)
path_anno = path+'annotations/Annotation/'

path_img = path+'images/Images'
labels = os.listdir(path_img)

print("No. of labels: {}".format(len(labels)))

print("-----------------")



for label in labels:

    print("{}, {} files".format(label, len(os.listdir(path_img+'/'+label))))
import matplotlib.pyplot as plt

from PIL import Image



fig, ax = plt.subplots(nrows=3, ncols=4,figsize=(20, 10))

fig.tight_layout()

cnt = 0

for row in ax:

    for col in row:

        image_name = np.random.choice(os.listdir(path_img+ '/' + labels[cnt]))

        im = Image.open(path_img+"/{}/{}".format(labels[cnt],image_name))

        col.imshow(im)

        col.set_title(labels[cnt])

        col.axis('off')

        cnt += 1

plt.show();
data = ImageDataBunch.from_folder(path_img, 

                                  ds_tfms=get_transforms(),

                                  valid_pct=0.2, 

                                  size=size, 

                                  bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(20,10))
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=[accuracy,error_rate], callback_fns=ShowGraph ,model_dir="/tmp/model/")
learn.model
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(6, figsize=(25,20));
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=10)
interp.plot_confusion_matrix(figsize=(40,40), dpi=60)
learn.fit_one_cycle(1)
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
data = ImageDataBunch.from_folder(path_img, 

                                  ds_tfms=get_transforms(),

                                  valid_pct=0.2, 

                                  size=299, 

                                  bs=bs//2).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=[accuracy,error_rate], callback_fns=ShowGraph ,model_dir="/tmp/model/")
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=10)
interp.plot_confusion_matrix(figsize=(40,40), dpi=60)
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=10)
interp.plot_confusion_matrix(figsize=(40,40), dpi=60)