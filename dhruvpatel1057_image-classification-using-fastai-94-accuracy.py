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
## importing all the libraries 
## here we will be building our model using fastai library

from fastai.vision import *
path = Path('../input/intel-image-classification/seg_train/seg_train/')  ## setting the path to the train images
path.ls()

classes =['glacier','sea','forest','street','mountain','buildings']  ## deifining labels of the images

for c in classes:
    print(c)
    verify_images(path/c,delete=True,max_size=500)
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
## here using ImageDataBunch object we are splitting data in to valid_pct and normalizing the images
data.classes   ## printing the classes of our dataset
data.show_batch(rows=3,figsize=(7,8))  ## viewing our data images
data.classes, data.c, len(data.train_ds),len(data.valid_ds)
model = cnn_learner(data,models.resnet34,metrics=error_rate)
model.fit_one_cycle(5)
model.model_dir ="/tmp/model/"


model.save('version-1')
model.unfreeze()
model.lr_find()
model.recorder.plot()
model.fit_one_cycle(2,max_lr=slice(1e-05,1e-04))
model.save("version 1.1")
interpret = ClassificationInterpretation.from_learner(model)
interpret.plot_confusion_matrix()
losess,idxs = interpret.top_losses()

interpret.plot_top_losses(9,figsize=(15,11))  ### these are some of our wrongly classified images
interpret.most_confused(min_val=5)   ## seeingw which were badly predictied
test_img = model.data.train_ds[0][0]
print(model.predict(test_img))

data.train_ds[0][0]


test_img = data.train_ds[3213][0]
print(model.predict(test_img))

data.train_ds[3213][0]

test_img = data.train_ds[6378][0]
print(model.predict(test_img))

data.train_ds[6378][0]
