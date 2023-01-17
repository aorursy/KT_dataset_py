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
# !pip install git+https://github.com/fastai/fastai2 
# from fastai2.vision.all import *

!pip install fastai
# import fastai
#https://docs.fast.ai/vision.models.xresnet
from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision.data import *
from fastai.vision.augment import *
from fastai.vision.learner import *
# from fastai.model import *
train = pd.read_csv('../input/super-ai-image-classification/train/train/train.csv')
test = pd.read_csv('../input/super-ai-image-classification/val/val/val.csv')

train.shape, test.shape
train_path = "../input/super-ai-image-classification/train/train/images"
test_path = "../input/super-ai-image-classification/val/val/images"
train.head()
item_tfms = [RandomResizedCrop(224, min_scale=0.75)]
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
def get_dls_from_df(df):
    df = df.copy()
    options = {
        "item_tfms": item_tfms,
        "batch_tfms": batch_tfms,
        "bs": 4,
    }
    dls = ImageDataLoaders.from_df(df, train_path, **options)
    return dls
dls = get_dls_from_df(train)
dls.show_batch()
from fastai.vision.all import *
# https://forums.fast.ai/t/problem-with-f1scoremulti-metric/63721
f1score = F1Score()
learn = cnn_learner(dls, resnet152, metrics=[f1score])
learn.fine_tune(2)
files = get_image_files(test_path)
file_list = []
pred_list = []
for file in files:
    pred = learn.predict(file)[2]
    pred = np.argmax(pred).item()
    pred_list.append(pred)
    file_list.append(file)
    
final_pred = pd.DataFrame({"id":file_list,'category':pred_list})
final_pred['category'].value_counts()
final_pred['id'] = final_pred['id'].astype(str)
final_pred['id'] = final_pred['id'].str.slice(54)
final_pred.to_csv('submission_resnet152_2cycle.csv',index = False)
# preds = learn.get_preds(files, with_decoded=True)
# preds
# preds
