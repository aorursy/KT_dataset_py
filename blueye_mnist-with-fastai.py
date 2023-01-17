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
%matplotlib inline

from fastai import *

from fastai.vision import *

import matplotlib
base = Path('../input')
data_df = pd.read_csv(base/'train.csv')

data_df.head()
val_df = data_df.sample(frac=0.2, random_state = 8456)

val_df.shape
trn_df = data_df.drop(val_df.index)

trn_df.shape
trn_x, trn_y = trn_df.loc[:, 'pixel0':'pixel783'], trn_df['label']

val_x, val_y = val_df.loc[:, 'pixel0':'pixel783'], val_df['label']
def save(path:Path, df, labels):

    path.mkdir(parents=True,exist_ok=True)

    unq_lbls=np.unique(labels)

    for label in unq_lbls:

        (path/str(label)).mkdir(parents=True,exist_ok=True)

    for i in range(len(df)):

        if(len(labels)!=0):

            matplotlib.image.imsave(str(path/str(labels[i])/(str(i) + '.jpg')), df[i])

        else:

            matplotlib.image.imsave(str(path/(str(i) + '.jpg')), df[i])
def reshape(dt_x, dt_y):

    dt_x = np.array(dt_x, dtype = np.uint8).reshape(-1,28,28)

    dt_x = np.stack((dt_x,)*3, axis=-1)

    dt_y = np.array(dt_y)

    return dt_x, dt_y
trn_x, trn_y = reshape(trn_x, trn_y)

val_x, val_y = reshape(val_x, val_y)
train=Path('../working/data/train')

save(train, trn_x, trn_y)

valid = Path('../working/data/valid')

save(valid, val_x, val_y)
path = Path('../working/data/')

data = (ImageList.from_folder(path)

        .split_by_folder(train='train', valid='valid')

        .label_from_folder()

        .transform(get_transforms(do_flip=False), size=28)

        .databunch(bs=256).normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, loss_func=nn.CrossEntropyLoss(), metrics=accuracy)
learn.fit_one_cycle(3, 1e-2)
learn.save('stage1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(15, slice(5e-5))
learn.save('stage2')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, 5e-5)
learn.save('stage3')

learn.load('stage3')

learn.freeze()
def create_tst(path:Path, test):

    path.mkdir(parents=True, exist_ok=True)

    for i in range(len(test)):

        matplotlib.image.imsave(str(path/(str(i) + '.jpeg')), test[i])
test_df = pd.read_csv(base/'test.csv')

test_df = np.array(test_df, dtype=np.uint8).reshape(-1,28,28)

test_df = np.stack((test_df,)*3, axis=-1)

test_df.shape
tst_path = Path('../working/test')

create_tst(tst_path, test_df)
preds = []

ImageId = []

for i in range(len(test_df)):

    img = open_image(tst_path/str(str(i)+'.jpeg'))

    pred_cls, pred_idx, pred_img = learn.predict(img)

    preds.append(int(pred_idx))

    ImageId.append(i+1)
submission = pd.DataFrame({'ImageId':ImageId, 'Label':preds})
submission.head()
submission.tail()
submission.to_csv('submission.csv',index=False)
import shutil

shutil.rmtree(tst_path)

path_val = Path('../working/data/valid')

shutil.rmtree(path_val)

path_trn = Path('../working/data/train')

shutil.rmtree(path_trn)