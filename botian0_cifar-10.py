# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install fastai2
from fastai2.vision.all import *
path = untar_data(URLs.CIFAR)
path.ls()
files = get_image_files(path/'train')
len(files)
test = get_image_files(path/'test')
len(test)
files[0],files[6000]
files[0].name
test[0]
dls = ImageDataLoaders.from_name_re(path, files, r'^\d+_(.*).png', item_tfms=Resize(224), batch_tfms=aug_transforms(size=224), add_test_folder=test)
dls.show_batch()
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.lr_find()
learn.fine_tune(2, 2.5e-3)
learn.show_results()
learn.predict(test[0])[0]
import re
file_re = r'^\d+_(.*).png'
x = re.search(file_re, test[0].name)
print(x.group(1))
def get_category(name):
    file_re = r'^\d+_(.*).png'
    x = re.search(file_re, name)
    return x.group(1)
preds = learn.get_preds()
len(preds[1])
preds[1]
learn.data.classes
