# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input/caltech256/256_objectcategories/256_ObjectCategories/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from pathlib import Path
from fastai.vision import *
from fastai.metrics import error_rate, accuracy
from fastai.callbacks.tracker import SaveModelCallback
def get_label(p:Path):
    return p.parent.name
base = Path(r"../input/caltech256/256_objectcategories/256_ObjectCategories/")
images_path_list =[p for p in base.rglob("*.jpg")]
data = ImageDataBunch.from_name_func(base, images_path_list, get_label,size=224,ds_tfms=get_transforms(),bs=64).normalize(imagenet_stats)
print(len(data.classes))
data.show_batch(rows=3, figsize=(7,6))
def get_learner(data,model_type="resnet34"):
    if model_type == 'resnet18':
        learner = cnn_learner(data, models.resnet18, metrics=accuracy)
    elif model_type =='resnet34':
        learner = cnn_learner(data, models.resnet34, metrics=accuracy)
    elif model_type == 'resnet50':
        learner = cnn_learner(data, models.resnet50, metrics=accuracy)
    learner.model_dir = "/kaggle/working"
    return learner
resnet18 = get_learner(data, 'resnet18')
resnet18.lr_find()
resnet18.recorder.plot()
resnet18.fit_one_cycle(5,slice(6e-2,1e-2),callbacks=[SaveModelCallback(resnet18, monitor='accuracy', mode='max',name='resnet18')])
resnet34 = get_learner(data, 'resnet34')
resnet34.lr_find()
resnet34.recorder.plot()
resnet34.fit_one_cycle(5,slice(6e-2,1e-2),callbacks=[SaveModelCallback(resnet34, monitor='accuracy', mode='max',name='resnet34')])
resnet50 = get_learner(data, 'resnet50')
resnet50.lr_find()
resnet50.recorder.plot()
resnet50.fit_one_cycle(5,slice(6e-2,1e-2),callbacks=[SaveModelCallback(resnet50, monitor='accuracy', mode='max',name='resnet50')])