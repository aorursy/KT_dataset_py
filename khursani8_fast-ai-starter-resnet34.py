# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
from fastai.vision import *

import sys

sys.path.insert(0, '../input/aptos2019-blindness-detection')
# copy pretrained weights for resnet34 to the folder fastai will search by default

Path('/tmp/.cache/torch/checkpoints/').mkdir(exist_ok=True, parents=True)

!cp '../input/resnet34/resnet34.pth' '/tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth'
PATH = Path('../input/aptos2019-blindness-detection')
df = pd.read_csv(PATH/'train.csv')

df.head()
df.diagnosis.value_counts()
%%time

# Try Oversampling



res = None

sample_to = df.diagnosis.value_counts().max()



for grp in df.groupby('diagnosis'):

    n = grp[1].shape[0]

    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)

    rows = pd.concat((grp[1], additional_rows))

    

    if res is None: res = rows

    else: res = pd.concat((res, rows))
res.diagnosis.value_counts()
src = (

    ImageList.from_df(res,PATH,folder='train_images',suffix='.png')

#         .use_partial_data(0.2)

        .split_by_rand_pct()

        .label_from_df()

    )

src
data = (

    src.transform(get_transforms(),size=128)

    .databunch()

    .normalize()

)

data
kappa = KappaScore()

kappa.weights = "quadratic"

loss_func = LabelSmoothingCrossEntropy()

learn = cnn_learner(data,models.resnet34,metrics=[accuracy,kappa],loss_func=loss_func,model_dir='/kaggle',pretrained=True).mixup()

# learn.model.cuda()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(1)
learn.fit_one_cycle(1,slice(1e-6,1e-3))
# # progressive resizing

learn.data = data = (

    src.transform(get_transforms(),size=224)

    .databunch()

    .normalize()

)

learn.freeze()

# # learn.lr_find()

# # learn.recorder.plot()
learn.fit_one_cycle(2,3e-4)
learn.unfreeze()

# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(1,slice(1e-6,3e-5))
sample_df = pd.read_csv(PATH/'sample_submission.csv')

sample_df.head()
learn.data.add_test(ImageList.from_df(sample_df,PATH,folder='test_images',suffix='.png'))
preds,y = learn.get_preds(DatasetType.Test)
sample_df.diagnosis = preds.argmax(1)

sample_df.head()
sample_df.to_csv('submission.csv',index=False)