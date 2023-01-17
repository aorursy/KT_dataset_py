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

from fastai import *

from fastai.vision import *

import torch
train = pd.read_csv('../input/train.csv')

test_file = pd.read_csv('../input/sample_submission.csv')

test_img = ImageList.from_df(test_file,path='../input/test',folder = 'test')



# Image augmentation on the images 

transforms = get_transforms(do_flip=True, flip_vert=True, max_rotate = 20.0, max_zoom=1.1, max_lighting = 0.5, max_warp = 0.2, p_affine = 0.75, p_lighting = 0.75)

train_img = (ImageList.from_df(train,path='../input/train',folder='train').split_by_rand_pct(0.01).label_from_df().add_test(test_img).transform(transforms,size=64)

             .databunch(path='.',bs=64,device=torch.device('cuda:0')).normalize(imagenet_stats))

#display training images



train_img.show_batch(rows=4, figsize=(10,10))
#Now we apply transfer learning where we use an already trained model by replacing its last layer and fitting it with the rest of the model.

learn = cnn_learner(train_img , models.densenet201, metrics = [error_rate,accuracy])
learn.lr_find()

#we use this magic function given in fastai to search for the best learning rate. More on this later
learn.recorder.plot(suggestion =True)
learn.recorder.plot()
alpha =  3e-02

learn.fit_one_cycle(5,slice(alpha))
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(7,6))
preds,_ = learn.get_preds(ds_type = DatasetType.Test)

# test_file.has_cactus = 
test_file.has_cactus = preds.numpy()[:,0]
test_file.to_csv('submission.csv',index=False)