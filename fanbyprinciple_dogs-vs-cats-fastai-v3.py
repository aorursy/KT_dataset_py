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
from fastai.vision import *
Config.data_path()
path = Config.data_path()/'dogs_vs_cats'
path.mkdir(parents=True, exist_ok=True)
path
# unzip test and train to path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if ('zip' in filename):
            file = os.path.join(dirname, filename)
            !unzip -q -n {file} -d {path}
!dir {path}
!dir {path}/'train'

# observation: we can use the first part of the file name before the dot to extract the labels
train_path = path/'train'
test_path = path/'test'
from fastai.metrics import error_rate
np.random.seed(42)
fnames = get_image_files(train_path)
fnames[:5]
categories = []
filenames = os.listdir(train_path)

for filename in filenames:
    if('dog.' in filename):
        categories.append(1)
    else:
        categories.append(0)

# 1 for dogs
# 0 for cats
import warnings
warnings.filterwarnings('ignore')
data = ImageDataBunch.from_lists(train_path, fnames, ds_tfms=get_transforms(), size=224, bs=64, labels=categories)
data.classes
data.show_batch(rows=3,figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
# PIL.UnidentifiedImageError: cannot identify image file '/root/.fastai/data/dogs_vs_cats/train/dog.2370.jpg'
learn.save('stage1-dogsvscats')
test_images = get_image_files(test_path)
test_images[:5]
submission = pd.DataFrame(os.listdir(test_path), columns=['ids'])
submission['label'] = 0
submission['id'] = 1
count = 0
for imgpath in test_images:
    img = open_image(imgpath)
    pred = learn.predict(img)
    if(str(pred[0]) != '1'):
        submission['label'][count] = 0
    else:
        submission['label'][count] = 1
    count = count +1

submission
count = 0
for imgpath in test_images:
    submission['id'][count] = count + 1
    count  = count + 1
submission[['id', 'label']].to_csv('submission.csv', index=False)
!dir /kaggle/working
#!mv outputs.csv /kaggle/working