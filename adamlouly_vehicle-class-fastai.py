# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.
from fastai import * 

from fastai.vision import *
path=Path('/kaggle/input/vehicle/train/train/')

ImageList.from_folder(path)
data = (ImageList.from_folder(path)

        #Where to find the data? 

        .split_by_rand_pct(0.20,seed=44)

        .label_from_folder()

        #Data augmentation? -> use tfms with a size of 128

        .transform(get_transforms(do_flip=True,flip_vert= False,max_zoom=1.1, max_lighting=0.2, max_warp=0.2),size=512)

        .databunch(bs=8)

        .normalize(imagenet_stats))  

print(data.classes)

data.show_batch(rows=3, figsize=(5,5))
Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp ../input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir='/kaggle',pretrained=True)

learn.summary()
#learn.lr_find()

#learn.recorder.plot()
defaults.device = torch.device('cuda') # makes sure the gpu is used

lr = 1e-2

learn.fit_one_cycle(5, slice(lr))
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6))
#load_test_data

submission = pd.read_csv("/kaggle/input/vehicle/sample_submission.csv")

testpath="/kaggle/input/vehicle/test/"

submission1 = submission

submission1['Id'] = submission['Id'].apply(lambda x: folder_name(x))

submission1

learn.data.add_test(ImageList.from_df(

    submission1, testpath,

    folder='testset',

    suffix='.jpg'

))
preds, _ = learn.get_preds(DatasetType.Test)

submission["Category"]=from_preds_to_list(preds)

submission.to_csv("submitnew.csv",index=False)
