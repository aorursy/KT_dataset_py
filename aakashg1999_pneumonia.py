# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/chest_xray/chest_xray"))



# Any results you write to the current directory are saved as output.
from fastai import*

from fastai.vision import*
path=Path("../input/chest_xray/chest_xray/test")
fnames = get_image_files(path/'PNEUMONIA') 

#fnames += get_image_files(path/'NORMAL')

fnames[:5]
img=fnames[-1]

f=open_image(img)

f.show(figsize=(5,5))

print(f.shape)

np.random.seed(2)

pat=re.compile(r'\d+_(\w+)_')
data = ImageDataBunch.from_name_re(path/'PNEUMONIA',fnames,pat,valid_pct=0.2, size=224,bs=5, num_workers=0

                                  ).normalize(imagenet_stats)
print(data.classes)

len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/tmp/models')
lr_find(learn)
learn.recorder.plot()
lr=1e-2
learn.fit_one_cycle(5,slice(lr))
learn.save('stage-1')

learn.fit_one_cycle(4,slice(8*1e-3))
learn.save('stage-2')
path2=Path("../input/chest_xray/chest_xray/val/PNEUMONIA")

fnames = get_image_files(path2)



i=0

try:

    while fnames[i]:

        img = open_image(fnames[i])

        classes=['bacteria', 'virus']

        data2 = ImageDataBunch.single_from_classes(path, classes, size=224).normalize(imagenet_stats)

        learn = cnn_learner(data2, models.resnet34, model_dir='/tmp/models').load('stage-2')

        pred_class,pred_idx,outputs = learn.predict(img)

        print(pred_class)

        img.show(figsize=(5,5))

        i=i+1

except IndexError:

    print('done')
learn.load('stage-1')

from IPython.display import FileLinks

FileLinks('/tmp')