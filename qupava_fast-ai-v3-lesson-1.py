# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *



bs = 64



path = untar_data(URLs.FLOWERS); path
path.ls()
path_img = path/'jpg'

fnames = get_image_files(path_img)

fnames[:5]
codes = np.loadtxt(path/'train.txt', dtype=str); codes
files_train = path/'train.txt'

files_validate = path/'valid.txt'

files_test = path/'test.txt'
train_array = pd.read_csv(files_train, header=None, delimiter=" ")

validate_array = pd.read_csv(files_validate, header=None, delimiter=" ")

test_array = pd.read_csv(files_test, header=None, delimiter=" ")



cat_array = pd.concat([train_array, validate_array, test_array])

data = ImageDataBunch.from_df(path=path, df=cat_array, size=224)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
#Training: resnet34

learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save("stage-1")
# Training Results

interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)

interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(18,18), dpi=60)

interp.most_confused(min_val=2)
# Unfreeze model and train more

learn.unfreeze()

learn.fit_one_cycle(1)
learn.load("stage-1")

learn.lr_find()

learn.recorder.plot()
# Still need to fine tune the model???

# learn.unfreeze()

# learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
# Training: resnet50
data = ImageDataBunch.from_df(path=path, df=cat_array, size=299, bs=bs//2)

learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8)
learn.save('stage-1-50')
learn.unfreeze()

# how should I fine tune the model here?

# learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)