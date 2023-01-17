import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))



from fastai.vision import *

from fastai.tabular import *

from fastai.metrics import error_rate

import csv

import numpy as np

import PIL 

import pandas as pd

#defaults.device = torch.device('cuda')
%reload_ext autoreload

%autoreload 2

%matplotlib inline
train = pd.read_csv('../input/train/train.csv')

train = train.drop(['Name','Breed2','Color3', 'Description'], axis=1)

train.head(2)
test = pd.read_csv('../input/test/test.csv'); 

test = test.drop(['Name','Breed2','Color3','Description'], axis=1)

test.head(2)
pet = test['PetID'].values

pred = []
cat_names = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'Vaccinated', 'Dewormed', 

             'Sterilized', 'Health', 'RescuerID','VideoAmt','PetID','PhotoAmt']

dep_var = 'AdoptionSpeed'
valid_idx = range(len(train)-1000, len(train))
procs = [FillMissing, Categorify, Normalize]

data = TabularDataBunch.from_df(path='../working/', df=train, dep_var=dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)

learn = tabular_learner(data, layers=[200,100], metrics=accuracy)

learn.fit_one_cycle(1, 1e-2)
predlist = []
for x in range(0,len(test)):

    pred.append(learn.predict(test.iloc[x]))
for x in range(0,len(test)):

    preds = pred[x][0]
for x in range(0,len(test)):

    predlist.append(pred[x][1].item())
submission = pd.DataFrame({'PetID':pet, 'AdoptionSpeed':predlist})

submission.head()
submission.to_csv('submission.csv', index=False)