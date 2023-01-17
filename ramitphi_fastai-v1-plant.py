!pip install fastai
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
from fastai.vision import *
#!pip list

data_path =Path('../input/plant-pathology-2020-fgvc7')
img_path = data_path/'images'
label_cols =['healthy','multiple_diseases','rust','scab']

train_df=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
train_df.head()
def give_label(row):
    for k,v in row[label_cols].items():
        if v==1:
            return k
train_df['labels']=train_df.apply(give_label,axis=1)
train_df['labels']
path=Path('/kaggle/input/plant-pathology-2020-fgvc7')
test1 = pd.read_csv(path/'test.csv')
test = (ImageList.from_df(test1,path,folder='images',suffix='.jpg',cols='image_id'))
test


data = ImageList.from_df(path=data_path,
                             df=train_df,
                              folder='images',
                              suffix='.jpg'
                             )

data = data.split_by_rand_pct(0.2)
data =data.label_from_df(cols='labels')
cols =data.classes
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
#!pip install "torch==1.4" "torchvision==0.5.0"
data = (data.transform(tfms, size=64).add_test(test).databunch(bs=64).normalize(imagenet_stats))
data
data.show_batch(rows=3, figsize=(12,9))
arch = models.resnet34
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, models.resnet50, metrics=error_rate,model_dir='/tmp/mod')

learn.fit_one_cycle(5)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(6 , max_lr=slice(1e-03,1e-02))

learn.save('plant1')
learn.export('/kaggle/working/plant1.pkl')
preds = learn.get_preds(DatasetType.Test)
test = pd.read_csv(path/'test.csv')
test_id = test['image_id'].values
submission = pd.DataFrame({'image_id': test_id})
submission = pd.concat([submission, pd.DataFrame(preds[0].numpy() , columns =cols)], axis=1)

submission.to_csv('submission_plantfirst.csv', index=False)
submission.head(10)
