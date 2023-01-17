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
# Put these at the top of every notebook, to get automatic reloading and inline plotting

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

# from fastai.model_selection import *

from fastai.metrics import error_rate

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tqdm import tqdm
PATH = "../input/Kannada-MNIST/"
# pd.read_csv(PATH+"train.csv").head(100)
train = pd.read_csv(PATH+"train.csv")

train.head()#, len(train)
test = pd.read_csv(PATH+"test.csv")

test.head()# len(test)
sample = pd.read_csv(PATH+"sample_submission.csv")
#train images and labels

image = train.iloc[:,1:]

label = train.iloc[:,0:1]
#test images and labels

test_image = test.iloc[:,1:]
test_id = test.iloc[:,0:1]

# test_id
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from PIL import Image
train['file_name'] = np.NaN
!mkdir train

!mkdir test

!ls train
#converting all pixels to image and saving it to to train folder

for i in tqdm(range(int(len(image)))):

    code = label.iloc[i,:].astype('int')[0]

    serial = i

    save_path = f'train/{serial}_train_{code}.png'

    train.iloc[i,785] = f'{serial}_train_{code}.png'

    

    temp_image = Image.fromarray(image.iloc[i,:].values.astype('uint8').reshape(28,28))

    temp_image.save(save_path)

    

    
test['file_name'] = np.NaN
#converting all pixels to image and saving it to to test folder

for i in tqdm(range(int(len(test_image)))):

    serial = i

    save_path = f'test/{serial}_test.png'

    test.iloc[i,785] = f'{serial}_test.png'

    

    temp_image = Image.fromarray(test_image.iloc[i,:].values.astype('uint8').reshape(28,28))

    temp_image.save(save_path)
train_dict = {'name':train['file_name'] , 'label': train['label']}

df = pd.DataFrame(train_dict)

df.head()
test_dict = {'name':test['file_name']}

df_test = pd.DataFrame(test_dict)

df_test.head()
tfms = get_transforms(do_flip=False)
src = (ImageList.from_df(path='train', df=df)

        .split_by_rand_pct()

        .label_from_df(cols='label')

       )
data = (src.transform(tfms, size=28)

        .databunch(bs=64).normalize(imagenet_stats)

       )
data.show_batch(rows=3)
!mkdir -p /tmp/.cache/torch/checkpoints

!cp /kaggle/input/fastai-pretrained-models/resnet50-19c8e357.pth /tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth



learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = Path('../kaggle/working'),path = Path("."))
# arch = models.resnet50

# # arch

# learn = cnn_learner(data, arch, metrics=[accuracy])
# learn
learn.lr_find()
learn.recorder.plot()
lr = 0.01

learn.fit_one_cycle(4, lr)
learn.save('kannada-mnist-stage-1')
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-4, lr/5))
learn.save('kanada-mnist-stage-2')
learn.export(file="/kaggle/working/export.pkl")
!ls
# pd.read_csv(PATH+"sample_submission.csv").head()
# train = ImageList.from_df(path='train', df=df)

# learn = load_learner(path="/kaggle/working", test=train)

# preds, _ = learn.get_preds(ds_type=DatasetType.Test)
# output = preds.argmax(dim=1)

# output[1:100]
dataframes = []

test = ImageList.from_df(path='test', df=df_test)

learn = load_learner(path="/kaggle/working/", test=test)

preds, _ = learn.get_preds(ds_type=DatasetType.Test)
output= preds.argmax(dim=1)
# np.unique(output)

# len(test_id), len(output)
df_sub = test_id
df_sub['label'] = pd.Series(output)
!rm -rf train
!rm -rf test
!ls
# df_sub.to_csv("submission.csv", index=False)
# submission = pd.DataFrame({ 'id': Id,

#                             'label': predictions })

df_sub.to_csv(path_or_buf ="submission.csv", index=False)
# pd.read_csv("submission.csv").head()
!ls