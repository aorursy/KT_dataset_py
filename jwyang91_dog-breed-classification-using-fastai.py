# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import fastai framework

from fastai import *

from fastai.vision import *

from torchvision.models import * 



import os

import matplotlib.pyplot as plt
#load data

path = Path("../input/stanford-dogs-dataset/")

path
path.ls()
# path_anno = path/'annotations/Annotations'

path_img = path/'images/Images/'



# path_anno

path_img
path_img.ls()

tfms = get_transforms()

# data = ImageDataBunch.from_folder(path_img ,train='.', valid_pct = 0.2,ds_tfms = tfms , size = 227)





np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=227, num_workers=0).normalize(imagenet_stats)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=227, num_workers=0, classes=data.classes[:100]).normalize(imagenet_stats)
data.show_batch(rows = 3 ,figsize = (7,6))

print(data.classes)

len(data.classes), data.c # data.c = for classification problems its number of classes
learn = create_cnn(data , models.resnet34, metrics = error_rate) 
learn.fit_one_cycle(4)
learn.model_dir = "/kaggle/working"

learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(3e-6,3e-5))

learn.model_dir = "/kaggle/working"

learn.save('stage-2')
learn.load('stage-2')

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9 , figsize = (15,11))
interp.plot_confusion_matrix(figsize = (30,30), dpi = 60)
interp.most_confused(min_val = 2) 
data.classes
img = open_image('../input/stanford-dogs-dataset/images/Images/n02105855-Shetland_sheepdog/n02105855_13382.jpg')

img
classes = data.classes

data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)

learn = create_cnn(data2 , models.resnet34)

learn.load('/kaggle/working/stage-2')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
prediction = str(pred_class)

prediction[10:]

print("The predicted breed is " + prediction[10:] + '.')