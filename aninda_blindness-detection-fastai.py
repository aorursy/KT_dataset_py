# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.vision import *

from fastai.metrics import error_rate



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input/aptos2019-blindness-detection"))
%reload_ext autoreload

%autoreload 2

%matplotlib inline
df_img=pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_img.head(5)

file = '../input/aptos2019-blindness-detection/train_images/'+df_img.iloc[0,0]+'.png'

print(file)

picture=plt.imread(file)

picture.shape
PATH = Path('../input/aptos2019-blindness-detection/')

data_dir = '../input/aptos2019-blindness-detection/'

train_df = pd.read_csv(os.path.join(data_dir,'train.csv'))

add_extension = lambda x: str(x) + '.png'

add_dir = lambda x: os.path.join('train_images', x)

train_df['id_code'] = train_df['id_code'].apply(add_extension)

train_df['id_code'] = train_df['id_code'].apply(add_dir)

print(train_df.shape)

data = (ImageList.from_df(train_df,PATH)

        #Where to find the data? 

        .split_by_rand_pct(0.20,seed=44)

        #How to split in train/valid? -> randomly with the default 20% in valid

        .label_from_df()

        #How to label? -> use the second column of the csv file and split the tags by ' '

        #Data augmentation? -> use tfms with a size of 128

        .transform(get_transforms(do_flip=True,flip_vert= True,max_zoom=1.1, max_lighting=0.2, max_warp=0.2, ),size=500)

        .databunch(bs=8)

        .normalize(imagenet_stats))                          

        #Finally -> use the defaults for conversion to databunch

data
print(data.classes)

data.show_batch(rows=3, figsize=(5,5))
!ls '../input/densenet201'
kappa = KappaScore()

kappa.weights = "quadratic"

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp ../input/densenet201/densenet201.pth /tmp/.cache/torch/checkpoints/densenet201-c1103571.pth



learn = cnn_learner(data, models.densenet201, metrics=[accuracy,kappa],model_dir='/kaggle',pretrained=True)

learn.summary()

learn.lr_find()

learn.recorder.plot()
lr = 1e-2

learn.fit_one_cycle(1, slice(lr))
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(10, slice(1e-6,1e-3))
learn.save("trained_model")
sample_df = pd.read_csv(PATH/'sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))

preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)