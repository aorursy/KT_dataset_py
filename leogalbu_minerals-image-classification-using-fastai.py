# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import warnings
warnings.filterwarnings("ignore")
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
from fastai import *
from fastai.vision import *
path = Path('/kaggle/input/minerals-identification-dataset/minet')
path
data_64 = (ImageList.from_folder(path) #have specified the train directory as it has a child dir named train which contains all the classes in folders
                .split_by_rand_pct(0.1, seed=33) #since there is no validation set, we are taking 10% of the train set as validation set
                .label_from_folder()#to label the images based on thier folder name/class
                .transform(get_transforms(), size=64)#using the default transforms and initial size of 64x64
                .databunch(bs=256)#batch size of 256, be cautious of OOM error when you increase the size of the image decrease the batchsize to be able to fit in the memory
                .normalize(imagenet_stats))#normalizing to the imagenet stats

data_64
learn = cnn_learner(data_64, #training on low res first 
                    models.resnet50, #loading the resenet18 arch with pretrained weights
                    metrics=accuracy, 
                    model_dir='/tmp/model/')
imageData = ImageDataBunch.from_folder(path, valid_pct=0.2, ds_tfms=get_transforms(flip_vert=True, max_rotate=45), size=250).normalize(imagenet_stats)
imageData.show_batch(5, figsize=(15, 11))
learn = cnn_learner(imageData, models.resnet50, pretrained=True, metrics=[accuracy, error_rate])
learn.model
learn.model_dir = "/kaggle/working"
learn.lr_find(start_lr=1e-07, num_it=100) 
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(4)
mingradlr = learn.recorder.min_grad_lr
print(mingradlr)
lr = mingradlr
learn.fit_one_cycle(10, lr)

learn.lr_find() 
learn.recorder.plot(suggestion=True)
learn.lr_find() 
learn.recorder.plot(suggestion=True)
mingradlr1 = learn.recorder.min_grad_lr
print(mingradlr1)
lr = 1e-04
learn.fit_one_cycle(10, lr)
learn.model_dir = "/kaggle/working"
learn.save("stage-1")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(10, figsize=(15,11))
interp.most_confused(min_val=2)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('stage-1');
learn.lr_find();
learn.recorder.plot(suggestion=True)
learn.unfreeze()
learn.fit_one_cycle(2, max_lr=1.91E-06)
#learn.fit_one_cycle(2, slice(1.91E-06, lr/10))
learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4)
interp.plot_confusion_matrix()